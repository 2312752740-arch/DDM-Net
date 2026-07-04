import os
import time
import argparse
import numpy as np
from PIL import Image
from glob import glob
from ntpath import basename
from os.path import join, exists
# pytorch libs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms as transforms
from model.MUnet import D_Model
from model.A import get_A
#data_EUVP/train/trainB
# ./Dataset/test/testA/## options../Mamba-UIE-HU/test_noref/U45/  ./Dataset_EUVP/test/testA/
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="./data_EUVP/test/testA/")#no_reference/UIEB60 
# parser.add_argument("--data_dir", type=str, default="./EUVP300/Araw/")

parser.add_argument("--sample_dir", type=str, default="./output/m13/EUVP215/")
parser.add_argument("--model_name", type=str, default="MuLA_GAN")
parser.add_argument("--weights_dir", type=str, default="./checkpoints_SAM/Y/")#checkpoint_evl ./checkpoints_SAM/m4/UIEB/
parser.add_argument("--T_sample", type=str, default="./output/m13/T/")


# parser.add_argument("--compare_dir", type=str, default="./output/14me_model/")
opt = parser.parse_args()

## checks
assert os.path.isdir(opt.weights_dir), "weights directory not found"
is_cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor

## model arch
# from models.D_net import Model
# from models.correct_color_net import Model

model = D_Model()


## data pipeline
img_width, img_height, channels = 256, 256, 3
transforms_ = [transforms.Resize((img_height, img_width), Image.BICUBIC),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]
transform = transforms.Compose(transforms_)

## testing loop for each weight
weights_files = sorted(glob(join(opt.weights_dir, "*.pth")))
for weights_path in weights_files:
    # Create a new directory for each weight
    weight_name = basename(weights_path)
    weight_sample_dir = join(opt.sample_dir, weight_name)
    os.makedirs(weight_sample_dir, exist_ok=True)
    T_dir=opt.T_sample
    os.makedirs(T_dir, exist_ok=True)


    model.load_state_dict(torch.load(weights_path, map_location=lambda storage, loc: storage))
    if is_cuda: model.cuda()
    model.eval()
    print("Loaded model from %s" % weights_path)

    times = []
    test_files = sorted(glob(join(opt.data_dir, "*.*")))
    for path in test_files:
        inp_img = transform(Image.open(path))
        inp_img = Variable(inp_img).type(Tensor).unsqueeze(0)
        s = time.time()
        gen_img= model(inp_img)
        # gen_img,t_map= model(inp_img)


        # gen_img,x_tb, x_td = model(inp_img)
        # a_out = get_A(inp_img).cuda()
        # I_rec = gen_img * x_td + (1 - x_tb) * a_out
        times.append(time.time() - s)
        # img_sample = torch.cat((inp_img.data, gen_img.data), -1)
        save_image(gen_img, join(weight_sample_dir,   basename(path)), normalize=True)
        # save_image(t_map, join(T_dir,  basename(path)), normalize=True)
        #join(opt.sample_dir, weight_name)
        print("Tested: %s with weights %s" % (path, weights_path))

    ## run-time
    if len(times) > 1:
        print("\nTotal samples: %d" % len(test_files))
        # accumulate frame processing times (without bootstrap)
        Ttime, Mtime = np.sum(times[1:]), np.mean(times[1:])
        print("Time taken: %d sec at %0.3f fps" % (Ttime, 1. / Mtime))
        print("Saved generated images in %s\n" % weight_sample_dir)
        #python evaluation/SSIM_PSNR.py  python test2.py  