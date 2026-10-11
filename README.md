# DDM-Net

Repository for **"Physics-Aware Underwater Image Enhancement via SAM-Guided Dual-Domain Mamba Network"** (submitted to *The Visual Computer*).

## Overview

DDM-Net is a physics-aware underwater image enhancement network that couples a Visual State Space Model (VSSM) for spatial long-range modeling with a Discrete Wavelet Transform (DWT) branch for frequency-domain detail restoration, under SAM-guided transmission-map supervision.

## Repository contents

| Path | Description |
|---|---|
| `pretrained/model_200.pth` | Pretrained weights (UIEB-800 + EUVP-1800, 256×256, 200 epochs) |
| `test.py` | Inference script (batch over a weights directory) |
| `test_var.py` | Inference script (single weight file, resize-256 evaluation protocol) |
| `train3_loss_all.py` | Training entry |
| `configs/train_paper.yaml` | Training configuration used in the paper |
| `model/A.py` | Physical prior: global atmospheric-light estimation |
| `utils/` | Data loading and loss functions |
| `tools/t_map.py` | SAM-guided transmission-map generation |
| `tools/t_map_udcp.py` | UDCP prior transmission-map generation |


## Environment

- Python 3.11, PyTorch 2.1.1 + CUDA 11.8, torchvision 0.16.1
- `mamba-ssm 1.2.0.post1` and `causal-conv1d 1.2.0.post1` (install from the official prebuilt wheels)
- Remaining dependencies: see `requirements.txt`
- `tools/t_map.py` additionally requires the `segment-anything` package and a SAM checkpoint (e.g. `sam_vit_h_4b8939.pth`) from https://github.com/facebookresearch/segment-anything

## Inference

```bash
python test_var.py \
  --model_module model.MUnet2 \
  --weight_path pretrained/model_200.pth \
  --root_data_dir <input_root> \
  --sample_dir <output_dir> \
  --img_size 256
```

`<input_root>` should contain per-dataset subfolders of raw images; enhanced results are written per subfolder under `<output_dir>`.

## Datasets

- UIEB: https://li-chongyi.github.io/proj_benchmark.html
- EUVP: https://irvlab.cs.umn.edu/resources/euvp-dataset
- OceanDark: https://github.com/tunai/l2uwe/tree/master/data
- U45: https://github.com/IPNUISTlegal/underwater-test-dataset-U45-

## Citation

If you find this work useful, please consider citing our paper once the publication information is available.
