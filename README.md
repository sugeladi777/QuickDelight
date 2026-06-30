# QuickDelight

QuickDelight 是一个从多视角人脸图片构建 UV partial-map，并用自监督重投影训练完整 UV 纹理图的人脸纹理补全项目。

## Current Flow

项目公开输入只有图片：

`multi-view images -> face mask -> Pixel3DMM image-space UV -> VGGT point map -> VGGTFace2 mesh -> partial-map -> self-supervised reprojection`

运行入口：

```bash
python -m quickdelight build-input <sample_id> \
  --raw-root data/ava256_flame_raw_16view_avif \
  --dataset-root data/quickdelight_dataset
```

批量构建：

```bash
python -m quickdelight build-dataset \
  --raw-root data/ava256_flame_raw_16view_avif \
  --dataset-root data/quickdelight_dataset
```

训练：

```bash
python -m quickdelight train-selfsup \
  --dataset-root data/quickdelight_dataset \
  --save-root data/runs/selfsup
```

`cache/vggtface2_input.pkl` 和 `vggtface2_mesh.obj` 是内部自动生成的中间结果，用于检查和调试；它们不是项目输入。

## Folder Layout

- `quickdelight/`: 当前主代码。
- `quickdelight/vggt_mesh/`: 图片输入构建、Pixel3DMM UV、VGGT point map、VGGTFace2 mesh 推理。
- `quickdelight/selfsup/`: 自监督训练、数据集和 loss。
- `quickdelight/input/`: 原始样本与相机 ID 的最小路径工具。
- `quickdelight/utils/`: 公共工具函数。
- `quickdelight/assets/`: 固定 UV mask。
- `data/ava256_flame_raw_16view_avif`: 保留的原始数据。
- `data/ava256_flame_raw_all_views_all_frames`: 保留的原始数据。
- `data/flame_mesh`: 保留的原始数据。
- `third_party/`: 外部依赖与模型代码。
