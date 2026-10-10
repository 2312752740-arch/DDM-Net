# print("Importing Libraries")
import datetime
import logging
import os
import sys

# import kornia
import yaml
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
# from scipy.cluster.hierarchy import weighted
from torchvision import datasets
from torchvision.models import VGG19_Weights
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
from tqdm import tqdm
# from models.loss import *
# from loss import *
from pytorch_msssim import MS_SSIM


# from models.commons import Weights_Normal, VGG19_PercepLoss, L2_L1_loss,TV_Loss,PerceptualLoss
# from models.commons import *

# from models.ME_model import ME_Generator, Discriminator
from model.MUnet2 import D_Model
# from models.D_net import net

from utils.data_utils import Dataloader
from utils.loss import Weights_Normal, VGG19_PercepLoss, L2_L1_loss, TV_Loss, PerceptualLoss

import time
from torchvision.transforms import InterpolationMode
from torchvision.models import VGG19_Weights
import matplotlib.pyplot as plt
from pytorch_msssim import ssim
from model.A import get_A
from utils.loss import Grad_loss



# 训练自己的数据集A_dataset2000，需要更改configs/train_MuLA-GAN.yaml，以及相应的dataloder

# 加入总变分损失

# 保存每次运行日志：
# print("Importing Done")
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", type=str, default="configs/train_models.yaml")
    parser.add_argument("--epoch", type=int, default=0, help="which epoch to start from")
    parser.add_argument("--num_epochs", type=int, default=201, help="number of epochs of training")  # default=301
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of 1st order momentum")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of 2nd order momentum")
    parser.add_argument("--method", default="b2", help="dataset name")

    parser.add_argument("--l1", type=float, default=1.0, help="weight for L1 loss")
    parser.add_argument("--per", type=float, default=0.5, help="weight for perceptual loss")
    parser.add_argument("--t", type=float, default=0.5, help="weight for transmission loss")
    parser.add_argument("--phy", type=float, default=0.2, help="weight for physical loss")
    return parser.parse_args()



def setup_logging(args, log_dir):
    # 确保日志文件夹存在，如果不存在则创建
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dataset_name = f"{current_time}_B{args.batch_size}_{args.method}"
    # dataset_name = f"20241105-151301_batch8_epoch80_lr0.0001"

    log_file_path = os.path.join(log_dir, f"{dataset_name}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )


# 新增函数：绘制损失曲线
def plot_losses(epochs, total_losses, args):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, total_losses, 'b-', label='Total Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 确保绘图目录存在
    plot_folder = "plots"
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    # 保存图像
    plot_path = os.path.join(plot_folder, f'loss_plot_{args.method}.png')
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Updated loss plot saved to {plot_path}")


# print("Reading done")
def main(args):
    log_dir = "checkpoints/log"  # 你想要保存日志文件的文件夹路径
    setup_logging(args, log_dir)

    ## training params
    epoch = args.epoch
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    lr_rate, lr_b1, lr_b2 = args.lr, args.b1, args.b2
    # load the data config file
    with open(args.cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # get info from config file
    dataset_name = cfg["dataset_name"]
    dataset_path = cfg["dataset_path"]
    img_width = cfg["im_width"]
    img_height = cfg["im_height"]
    ckpt_interval = cfg["ckpt_interval"]

    # print("##############")
    # create dir for model and validation data
    checkpoint_dir = os.path.join("checkpoints_SAM/", args.method)
    os.makedirs(checkpoint_dir, exist_ok=True)

    tv_loss = TV_Loss()
    # l1_l2 = L2_L1_loss()
    L_vgg = VGG19_PercepLoss(weights=VGG19_Weights.IMAGENET1K_V1)
    mse_loss = torch.nn.MSELoss()
    L1_loss=torch.nn.L1Loss()
    ms_ssim_loss = MS_SSIM(data_range=1.0, size_average=True, channel=3)
    L_grad=Grad_loss()


    model = D_Model()
    # 初始化用于记录损失的列表
    total_loss_history = []
    epochs_history = []

    # see if cuda is available
    if torch.cuda.is_available():

        model = model.cuda()
        # tv_loss = tv_loss.cuda()
        # l1_l2 = l1_l2.cuda()
        L_vgg = L_vgg.cuda()
        mse_loss = mse_loss.cuda()
        L1_loss=L1_loss.cuda()
        ms_ssim_loss = ms_ssim_loss.cuda() #
        L_grad=L_grad.cuda()


        Tensor = torch.cuda.FloatTensor

    else:
        Tensor = torch.FloatTensor
    # Optimizers
    optimizer_G = torch.optim.Adam(model.parameters(), lr=lr_rate, betas=(lr_b1, lr_b2))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=num_epochs, eta_min=1e-6)


    ## Data pipeline
    transforms_ = [
        transforms.Resize((img_height, img_width), InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    # dataset_path=./Dataset/
    dataloader = DataLoader(
        Dataloader(dataset_path, dataset_name, transforms_=transforms_),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    logging.info(f"Number of batches per epoch: {len(dataloader)}")

    # 使用命令行传入的参数
    lambda_l1 = args.l1
    lambda_per = args.per
    lambda_t = args.t
    lambda_phy = args.phy
    lambda_ssim_weight = 1.0 

    logging.info(f"Loss weights: rec={lambda_l1}, per={lambda_per}, phy={lambda_phy}, t={lambda_t}")

    if epoch > 0:
        # Load model checkpoints
        model_path = os.path.join(checkpoint_dir, f"best_model.pth")
        if os.path.isfile(model_path):
            model.load_state_dict(torch.load(model_path))
            logging.info(f"Loaded model checkpoint for epoch {epoch}")
        else:
            logging.info(f"No checkpoint found for epoch {epoch}. Starting training from scratch.")

    best_loss_G = float('inf')  # 在开始循环的时候设定
    ## Training pipeline
    for epoch in range(epoch, num_epochs):
        start_time = time.time()
        epoch_total_loss = 0.0  # 用于累积每个epoch的总损失

        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), smoothing=0.9):
            # Model inputs
            imgs_distorted = Variable(batch["A"].type(Tensor))
            imgs_good_gt = Variable(batch["B"].type(Tensor))
            t_maps_sam = Variable(batch["T"].type(Tensor))     # SAM引导的传输图

            optimizer_G.zero_grad()
            # J_pred,t_maps= model(imgs_distorted)
            J_pred,t_pred= model(imgs_distorted)


            B = get_A(imgs_distorted).cuda()
            I_rec = J_pred * t_pred + B * (1 - t_pred)

            J_pred_01 = (J_pred + 1) / 2
            imgs_good_gt_01 = (imgs_good_gt + 1) / 2
            loss_ms_ssim = 1 - ms_ssim_loss(J_pred_01, imgs_good_gt_01)
            




            #重建损失
            # loss_ssim0 = 1 - torch.mean(ssim(J_pred, imgs_good_gt)).cuda()
            l1_loss=L1_loss(J_pred, imgs_good_gt)
            L_rec=l1_loss+lambda_ssim_weight*loss_ms_ssim

            #t-损失
            loss_trans_value = L1_loss(t_pred, t_maps_sam) # 数值稍微像一点就行
            loss_trans_edge = L_grad(t_pred, t_maps_sam) # 边缘必须非常像！
            loss_trans_guide = 0.5 * loss_trans_value + 1.5 * loss_trans_edge # 增加边缘权中
            #物理约束损失
            l_phy = L1_loss(I_rec, imgs_distorted)

            #感知损失：
            l_per=L_vgg(J_pred, imgs_good_gt)
            


            # 总损失函数
            loss_G = L_rec * lambda_l1 + l_per * lambda_per + l_phy * lambda_phy + loss_trans_guide * lambda_t


            loss_G.backward()
            optimizer_G.step()
            epoch_total_loss += loss_G.item()  # 累积每个batch的损失
        # 计算该epoch的平均损失
        epoch_avg_loss = epoch_total_loss / len(dataloader)
        total_loss_history.append(epoch_avg_loss)
        epochs_history.append(epoch)
        scheduler.step()
        logging.info(f"Current learning rate: {optimizer_G.param_groups[0]['lr']:.8f}")


        # current_lr = optimizer_G.param_groups[0]['lr']
        # logging.info(f"Current learning rate: {current_lr:.6f}")


        logging.info(
            f"[ GLoss: {loss_G.item():.3f} || L_rec:{L_rec.item():.3f} || l_per:{l_per.item():.3f} || l_phy:{l_phy.item():.3f} || L_t:{loss_trans_guide.item():.3f} ]")#
        # 记录时间
        end_time = time.time()
        epoch_time = end_time - start_time
        logging.info(f"\nEpoch{epoch} took {epoch_time: .2f} seconds to complete")
        logging.info(f"[Epoch {epoch} Average GLoss: {epoch_avg_loss:.3f} || ... ]")

        ## Save model checkpoints
        # 保存最好的模型
        if epoch_avg_loss < best_loss_G:
            best_loss_G = epoch_avg_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
            logging.info(f"Best model saved at epoch {epoch} with loss_G: {best_loss_G:.3f}")

        # 定期保存检查点（不包含 best）
        if epoch % 10 == 0 and epoch >= 150:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_{epoch}.pth"))
            logging.info(f"Checkpoint saved at epoch {epoch}")


        plot_losses(range(len(total_loss_history)), total_loss_history, args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
# python train2.py
