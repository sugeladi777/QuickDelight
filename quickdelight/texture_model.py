from __future__ import annotations

"""Modern multi-view UV texture completion baseline.

The model follows the current project direction: fuse several observed UV
partial-maps, predict a complete coarse texture, then refine it in UV space.
It intentionally avoids heavyweight diffusion dependencies so it can be used as
the first reproducible baseline for self-supervised training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _make_group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = min(max_groups, num_channels)
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class ConvNeXtBlock(nn.Module):
    """Small ConvNeXt-style residual block for UV-space completion."""

    def __init__(self, channels: int, expansion: int = 4):
        super().__init__()
        hidden = channels * expansion
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm = _make_group_norm(channels)
        self.pointwise = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), 1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gamma * self.pointwise(self.norm(self.depthwise(x)))


class MaskAwareStem(nn.Module):
    """Encode RGB plus visibility mask without treating missing texels as real black."""

    def __init__(self, in_channels: int, out_channels: int, use_mask: bool):
        super().__init__()
        self.use_mask = use_mask
        channels = in_channels + 1 if use_mask else in_channels
        self.proj = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=5, padding=2, bias=False),
            _make_group_norm(out_channels),
            nn.GELU(),
            ConvNeXtBlock(out_channels),
        )

    def forward(self, image: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if self.use_mask:
            if mask is None:
                raise ValueError("masks must be provided when use_mask=True")
            image = image * mask
            image = torch.cat([image, mask], dim=1)
        return self.proj(image)


class DownStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, depth: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            _make_group_norm(out_channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*(ConvNeXtBlock(out_channels) for _ in range(depth)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.down(x))


class UpStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, depth: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=1, bias=False),
            _make_group_norm(out_channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*(ConvNeXtBlock(out_channels) for _ in range(depth)))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.blocks(self.proj(torch.cat([x, skip], dim=1)))


class MultiViewEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, use_mask: bool, use_checkpoint: bool):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.stem = MaskAwareStem(in_channels, c1, use_mask=use_mask)
        self.down1 = DownStage(c1, c2, depth=2)
        self.down2 = DownStage(c2, c3, depth=2)
        self.down3 = DownStage(c3, c4, depth=3)

    def forward(self, images: torch.Tensor, masks: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        f1 = self._run(self.stem, images, masks)
        f2 = self._run(self.down1, f1)
        f3 = self._run(self.down2, f2)
        f4 = self._run(self.down3, f3)
        return f1, f2, f3, f4

    def _run(self, module: nn.Module, *inputs: torch.Tensor | None) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            tensor_inputs = tuple(item for item in inputs if item is not None)
            if len(tensor_inputs) == len(inputs):
                return checkpoint(module, *tensor_inputs, use_reentrant=False)
        return module(*inputs)


class ConfidenceFusion(nn.Module):
    """Fuse view features with learned confidence and observed-mask gating."""

    def __init__(self, channels: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(channels, max(1, channels // 2), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(max(1, channels // 2), 1, kernel_size=1),
        )

    def forward(self, features: torch.Tensor, masks: torch.Tensor | None, batch_size: int, view_count: int) -> torch.Tensor:
        _, channels, height, width = features.shape
        features = features.view(batch_size, view_count, channels, height, width)
        confidence = self.head(features.reshape(batch_size * view_count, channels, height, width))
        confidence = confidence.view(batch_size, view_count, 1, height, width)
        if masks is not None:
            masks_down = masks.reshape(batch_size * view_count, 1, masks.shape[-2], masks.shape[-1]).float()
            masks_down = F.interpolate(masks_down, size=(height, width), mode="nearest")
            masks_down = masks_down.view(batch_size, view_count, 1, height, width)
            confidence = confidence.masked_fill(masks_down <= 0, -1e4)
        weights = torch.softmax(confidence, dim=1)
        return (features * weights).sum(dim=1)


class CoarseToRefineDecoder(nn.Module):
    def __init__(self, base_channels: int, out_channels: int, use_checkpoint: bool):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.bottleneck = nn.Sequential(*(ConvNeXtBlock(c4) for _ in range(4)))
        self.up3 = UpStage(c4, c3, c3, depth=2)
        self.up2 = UpStage(c3, c2, c2, depth=2)
        self.up1 = UpStage(c2, c1, c1, depth=2)
        self.coarse_head = nn.Conv2d(c1, out_channels, kernel_size=1)
        self.refine = nn.Sequential(
            nn.Conv2d(c1 + out_channels + 1, c1, kernel_size=3, padding=1, bias=False),
            _make_group_norm(c1),
            nn.GELU(),
            ConvNeXtBlock(c1),
            ConvNeXtBlock(c1),
            nn.Conv2d(c1, out_channels, kernel_size=1),
        )

    def forward(self, f1: torch.Tensor, f2: torch.Tensor, f3: torch.Tensor, f4: torch.Tensor, fused_mask: torch.Tensor) -> torch.Tensor:
        x = self._run(self.bottleneck, f4)
        x = self._run(self.up3, x, f3)
        x = self._run(self.up2, x, f2)
        x = self._run(self.up1, x, f1)
        coarse = torch.sigmoid(self.coarse_head(x))
        residual = self._run(self.refine, torch.cat([x, coarse, fused_mask], dim=1))
        return torch.clamp(coarse + 0.1 * torch.tanh(residual), 0.0, 1.0)

    def _run(self, module: nn.Module, *inputs: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return checkpoint(module, *inputs, use_reentrant=False)
        return module(*inputs)


class MultiViewUVCompletionNet(nn.Module):
    """FreeUV/MV2UV-inspired feed-forward baseline for partial UV completion."""

    def __init__(
        self,
        in_channels: int = 3,
        use_mask: bool = True,
        base_channels: int = 32,
        out_channels: int = 3,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.use_mask = use_mask
        self.encoder = MultiViewEncoder(in_channels, base_channels, use_mask=use_mask, use_checkpoint=use_checkpoint)
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.fusions = nn.ModuleList(ConfidenceFusion(channel) for channel in channels)
        self.decoder = CoarseToRefineDecoder(base_channels, out_channels, use_checkpoint=use_checkpoint)

    def forward(self, inputs: torch.Tensor, masks: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, view_count, channels, height, width = inputs.shape
        flat_inputs = inputs.reshape(batch_size * view_count, channels, height, width)
        flat_masks = masks.reshape(batch_size * view_count, 1, height, width) if masks is not None else None
        features = self.encoder(flat_inputs, flat_masks)
        fused = [
            fusion(feature, masks, batch_size, view_count)
            for fusion, feature in zip(self.fusions, features)
        ]
        fused_mask = self._fuse_mask(masks, batch_size, view_count, height, width, inputs.device, inputs.dtype)
        return self.decoder(fused[0], fused[1], fused[2], fused[3], fused_mask)

    @staticmethod
    def _fuse_mask(
        masks: torch.Tensor | None,
        batch_size: int,
        view_count: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if masks is None:
            return torch.ones((batch_size, 1, height, width), device=device, dtype=dtype)
        return masks.float().amax(dim=1)


class TextureCompletionNet(nn.Module):
    def __init__(self, base_channels: int = 32, use_mask: bool = True, use_checkpoint: bool = True):
        super().__init__()
        self.network = MultiViewUVCompletionNet(
            in_channels=3,
            use_mask=use_mask,
            base_channels=base_channels,
            out_channels=3,
            use_checkpoint=use_checkpoint,
        )

    def forward(self, inputs: torch.Tensor, masks: torch.Tensor | None = None) -> torch.Tensor:
        return self.network(inputs, masks)
