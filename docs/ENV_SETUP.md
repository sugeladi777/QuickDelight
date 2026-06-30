# QuickDelight 环境搭建

本文档记录 `/home1/lichengkai/QuickDelight` 在新服务器上的实际可用搭建流程。

## 1. 基础信息

- 项目路径：`/home1/lichengkai/QuickDelight`
- Conda 根目录：`/home/lichengkai/anaconda3`
- 环境名：`quickdelight`
- 显卡：RTX 3090
- 已配置用户级镜像：
  - `conda`：`defaults / conda-forge / pytorch` 走清华，`nvidia` 走官方
  - `pip`：清华源

## 2. 创建新环境

```bash
conda create -y -n quickdelight python=3.10
conda activate quickdelight
```

## 3. 安装 PyTorch

项目当前按 `torch 2.0.0 + cu118` 验证通过：

```bash
python -m pip install \
  torch==2.0.0+cu118 \
  torchvision==0.15.0+cu118 \
  torchaudio==2.0.0+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install numpy==1.26.4
```

## 4. 安装主依赖

```bash
python -m pip install \
  pillow \
  tqdm \
  opencv-python==4.11.0.86 \
  scipy \
  tensorboard \
  trimesh \
  pyyaml \
  huggingface_hub==0.33.1 \
  einops==0.8.1 \
  omegaconf==2.3.0 \
  timm==0.9.16 \
  pytorch-lightning==2.0.9 \
  ninja \
  scikit-image \
  environs \
  loguru \
  yacs \
  mediapy \
  tyro \
  distinctipy \
  validators \
  wandb \
  h5py \
  face-alignment==1.3.3
python -m pip install --upgrade 'setuptools<81' wheel
```

说明：

- `setuptools<81` 是为了保证 `pkg_resources` 可用，否则 `torch.utils.cpp_extension` 会报错，影响 `nvdiffrast` 编译。

## 5. 安装第三方源码依赖

### 5.1 Pixel3DMM

```bash
python -m pip install -e /home1/lichengkai/QuickDelight/third_party/pixel3dmm --no-deps
```

### 5.2 人脸检测与解析

注意导入名是 `facer`，但安装包名是 `pyfacer`：

```bash
python -m pip install https://codeload.github.com/FacePerceiver/facer/tar.gz/refs/heads/main
python -m pip install https://codeload.github.com/elliottzheng/batch-face/tar.gz/refs/heads/master
```

### 5.3 nvdiffrast

本机需要显式使用 CUDA 11.7 编译，不能直接用默认 `pip install git+...`。

```bash
export CUDA_HOME=/usr/local/cuda-11.7
export PATH=/usr/local/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST=8.6

tmpdir=/tmp/nvdiffrast-main-src
rm -rf "$tmpdir"
mkdir -p "$tmpdir"
cd "$tmpdir"
curl -L https://codeload.github.com/NVlabs/nvdiffrast/tar.gz/refs/heads/main | tar -xz
cd "$tmpdir/nvdiffrast-main"
/home/lichengkai/anaconda3/envs/quickdelight/bin/python setup.py install
```

## 6. 导入检查

```bash
python - <<'PY'
mods = [
    'torch', 'torchvision', 'torchaudio', 'numpy', 'cv2', 'trimesh',
    'timm', 'omegaconf', 'einops', 'huggingface_hub', 'pytorch_lightning',
    'pixel3dmm', 'facer', 'batch_face'
]
for name in mods:
    __import__(name)
    print(name, 'OK')
import nvdiffrast.torch as dr
print('nvdiffrast OK')
PY
```

## 7. QuickDelight smoke test

已验证可跑通的单样本命令：

```bash
export CUDA_HOME=/usr/local/cuda-11.7
export PATH=/usr/local/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:${LD_LIBRARY_PATH:-}

python -m quickdelight build-input 20210810--1306--FXN596_031440 \
  --raw-root /home1/lichengkai/QuickDelight/data/ava256_flame_raw_16view_avif \
  --dataset-root /home1/lichengkai/QuickDelight/data/quickdelight_env_smoke \
  --views 2 \
  --image-size 512 \
  --texture-size 1024 \
  --device cuda:0 \
  --overwrite
```

成功后会生成：

- `cache/crop_metadata.json`
- `cache/vggtface2_input.pkl`
- `vggtface2_mesh.obj`
- `input/partial_quality.json`

## 8. 当前代码侧已处理的兼容项

- `quickdelight/vggt_mesh/image_pipeline.py` 会自动补 `PIXEL3DMM_CODE_BASE / PREPROCESSED_DATA / TRACKING_OUTPUT` 默认值，不再要求手工写 `.env`。
- `third_party/pixel3dmm/src/pixel3dmm/lightning/p3dmm_network.py` 已做新服务器兼容修复，避免 `torch 2.0.0` 下 native MHA fast path 报错。
