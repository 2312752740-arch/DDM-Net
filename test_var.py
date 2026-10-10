import argparse
import importlib
import os
import time
from glob import glob
from os.path import basename, isdir, join, normpath

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import InterpolationMode
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument("--model_module", type=str, required=True)
parser.add_argument("--weight_path", type=str, required=True)
parser.add_argument("--root_data_dir", type=str, default="./eval_data/")
parser.add_argument("--sample_dir", type=str, required=True)
parser.add_argument("--img_size", type=int, default=256)
opt = parser.parse_args()

mod = importlib.import_module(opt.model_module)
D_Model = getattr(mod, "D_Model")

assert os.path.isdir(opt.root_data_dir), "root data dir not found"
assert os.path.isfile(opt.weight_path), "weight not found"

data_dirs = sorted([join(opt.root_data_dir, d) for d in os.listdir(opt.root_data_dir)
                    if isdir(join(opt.root_data_dir, d))])
assert len(data_dirs) > 0

is_cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor

model = D_Model()
model.load_state_dict(torch.load(opt.weight_path, map_location=lambda s, loc: s))
if is_cuda:
    model.cuda()
model.eval()
weight_name = basename(opt.weight_path).replace(".pth", "")
print(f"Loaded {opt.weight_path} into {opt.model_module}")

tf = transforms.Compose([
    transforms.Resize((opt.img_size, opt.img_size), InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

for data_dir in data_dirs:
    dataset_name = os.path.basename(normpath(data_dir))
    result_dir = join(opt.sample_dir, weight_name, dataset_name)
    os.makedirs(result_dir, exist_ok=True)
    files = sorted([p for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp")
                    for p in glob(join(data_dir, ext))])
    if not files:
        print(f"Warning: no images in {data_dir}, skip")
        continue
    times = []
    with torch.no_grad():
        for path in files:
            inp = Variable(tf(Image.open(path).convert("RGB"))).type(Tensor).unsqueeze(0)
            s = time.time()
            out = model(inp)
            gen = out[0] if isinstance(out, (tuple, list)) else out
            times.append(time.time() - s)
            save_image(gen, join(result_dir, basename(path)), normalize=True)
    print(f"{dataset_name}: {len(files)} imgs, avg {1000*sum(times)/len(times):.1f} ms -> {result_dir}")

print("ALL DONE")
