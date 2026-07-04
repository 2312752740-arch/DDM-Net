# DDM-Net

Official inference code and pretrained weights for **"Physics-Aware Underwater Image Enhancement via SAM-Guided Dual-Domain Mamba Network"** (The Visual Computer, 2026).

## Quick Start

```bash
pip install torch torchvision
python test.py --input input.png --output output.png --weights pretrained/best_model.pth
