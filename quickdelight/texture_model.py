from __future__ import annotations

"""Multi-view UV texture completion model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _make_group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = min(max_groups, num_channels)
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = _make_group_norm(out_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = _make_group_norm(out_channels)
        self.act2 = nn.ReLU(inplace=True)
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), _make_group_norm(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act2(out + identity)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = _make_group_norm(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.block = ResidualConvBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.act(self.norm(self.down(x))))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = ResidualConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class EncoderDeep(nn.Module):
    def __init__(self, in_channels: int = 4, base_channels: int = 32, use_checkpoint: bool = True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.stem = ResidualConvBlock(in_channels, c1)
        self.down1 = DownBlock(c1, c2)
        self.down2 = DownBlock(c2, c3)
        self.down3 = DownBlock(c3, c4)
        self.bottleneck = ResidualConvBlock(c4, c4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        f1 = self._run(self.stem, x)
        f2 = self._run(self.down1, f1)
        f3 = self._run(self.down2, f2)
        f4 = self._run(self.down3, f3)
        return f1, f2, f3, self._run(self.bottleneck, f4)

    def _run(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return checkpoint(module, x, use_reentrant=False)
        return module(x)


class DecoderDeep(nn.Module):
    def __init__(self, base_channels: int = 32, out_channels: int = 3, use_checkpoint: bool = True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.up3 = UpBlock(c4, c3, c3)
        self.up2 = UpBlock(c3, c2, c2)
        self.up1 = UpBlock(c2, c1, c1)
        self.head = nn.Sequential(ResidualConvBlock(c1, c1), nn.Conv2d(c1, out_channels, kernel_size=1))

    def forward(self, f1: torch.Tensor, f2: torch.Tensor, f3: torch.Tensor, f4: torch.Tensor) -> torch.Tensor:
        x = self._run(self.up3, f4, f3)
        x = self._run(self.up2, x, f2)
        x = self._run(self.up1, x, f1)
        return torch.sigmoid(self._run(self.head, x))

    def _run(self, module: nn.Module, *inputs: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return checkpoint(module, *inputs, use_reentrant=False)
        return module(*inputs)


class MultiViewFusionNetDeep(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        use_mask: bool = True,
        base_channels: int = 32,
        out_channels: int = 3,
        eps: float = 1e-6,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.use_mask = use_mask
        self.eps = eps
        encoder_in_channels = in_channels + 1 if use_mask else in_channels
        self.encoder = EncoderDeep(in_channels=encoder_in_channels, base_channels=base_channels, use_checkpoint=use_checkpoint)
        self.decoder = DecoderDeep(base_channels=base_channels, out_channels=out_channels, use_checkpoint=use_checkpoint)
        feature_channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.confidence_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, max(1, channels // 2), kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(max(1, channels // 2), 1, kernel_size=1),
                )
                for channels in feature_channels
            ]
        )

    def _fuse_scale(self, features: torch.Tensor, masks: torch.Tensor | None, batch_size: int, view_count: int, head: nn.Module) -> torch.Tensor:
        _, channels, height, width = features.shape
        features = features.view(batch_size, view_count, channels, height, width)
        confidence = head(features.reshape(batch_size * view_count, channels, height, width)).view(batch_size, view_count, 1, height, width)
        if masks is None:
            weights = torch.softmax(confidence, dim=1)
            return (features * weights).sum(dim=1)

        masks_down = masks.reshape(batch_size * view_count, 1, masks.shape[-2], masks.shape[-1]).float()
        masks_down = F.interpolate(masks_down, size=(height, width), mode="nearest").view(batch_size, view_count, 1, height, width)
        weights = torch.softmax(confidence.masked_fill(masks_down <= 0, -1e4), dim=1)
        return (features * weights).sum(dim=1)

    def forward(self, inputs: torch.Tensor, masks: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, view_count, _, height, width = inputs.shape
        if self.use_mask:
            if masks is None:
                raise ValueError("masks must be provided when use_mask=True")
            encoder_inputs = torch.cat([inputs, masks], dim=2)
        else:
            encoder_inputs = inputs
        encoder_inputs = encoder_inputs.reshape(batch_size * view_count, encoder_inputs.shape[2], height, width)
        f1, f2, f3, f4 = self.encoder(encoder_inputs)
        return self.decoder(
            self._fuse_scale(f1, masks, batch_size, view_count, self.confidence_heads[0]),
            self._fuse_scale(f2, masks, batch_size, view_count, self.confidence_heads[1]),
            self._fuse_scale(f3, masks, batch_size, view_count, self.confidence_heads[2]),
            self._fuse_scale(f4, masks, batch_size, view_count, self.confidence_heads[3]),
        )


class TextureCompletionNet(nn.Module):
    def __init__(self, base_channels: int = 32, use_mask: bool = True, use_checkpoint: bool = True):
        super().__init__()
        self.network = MultiViewFusionNetDeep(
            in_channels=3,
            use_mask=use_mask,
            base_channels=base_channels,
            out_channels=3,
            use_checkpoint=use_checkpoint,
        )

    def forward(self, inputs: torch.Tensor, masks: torch.Tensor | None = None) -> torch.Tensor:
        return self.network(inputs, masks)
