"""
 > Common/standard network archutectures and modules
 > Credit for some functions
    * github.com/eriklindernoren/PyTorch-GAN
    * pluralsight.com/guides/artistic-neural-style-transfer-with-pytorch
 > Maintainer: https://github.com/xahidbuffon
"""
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np

from torchvision.models import vgg16



def Weights_Normal(m):
    # initialize weights as Normal(mean, std)
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class UNetDown(nn.Module):
    """ Standard UNet down-sampling block 
    """
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """ Standard UNet up-sampling block
    """
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


# class VGG19_PercepLoss(nn.Module):
#     """ Calculates perceptual loss in vgg19 space
#     """
#     def __init__(self, _pretrained_=True):
#         super(VGG19_PercepLoss, self).__init__()
#         self.vgg = models.vgg19(pretrained=_pretrained_).features
#         for param in self.vgg.parameters():
#             param.requires_grad_(False)
#
#     def get_features(self, image, layers=None):
#         if layers is None:
#             layers = {'30': 'conv5_2'} # may add other layers
#         features = {}
#         x = image
#         for name, layer in self.vgg._modules.items():
#             x = layer(x)
#             if name in layers:
#                 features[layers[name]] = x
#         return features
#
#     def forward(self, pred, true, layer='conv5_2'):
#         true_f = self.get_features(true)
#         pred_f = self.get_features(pred)
#         return torch.mean((true_f[layer]-pred_f[layer])**2)
from torchvision.models import vgg19, VGG19_Weights
import torch.nn as nn

class VGG19_PercepLoss(nn.Module):
    """ Calculates perceptual loss in vgg19 space
    """
    def __init__(self, weights=VGG19_Weights.IMAGENET1K_V1):
        super(VGG19_PercepLoss, self).__init__()
        self.vgg = vgg19(weights=weights).features
        for param in self.vgg.parameters():
            param.requires_grad_(False)

    def get_features(self, image, layers=None):
        if layers is None:
            layers = {'30': 'conv5_2'} # may add other layers
        features = {}
        x = image
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

    def forward(self, pred, true, layer='conv5_2'):
        true_f = self.get_features(true)
        pred_f = self.get_features(pred)
        return torch.mean((true_f[layer] - pred_f[layer]) ** 2)



class Gradient_Penalty(nn.Module):
    """ Calculates the gradient penalty loss for WGAN GP
    """
    def __init__(self, cuda=True):
        super(Gradient_Penalty, self).__init__()
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    def forward(self, D, real, fake):
        # Random weight term for interpolation between real and fake samples
        eps = self.Tensor(np.random.random((real.size(0), 1, 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (eps * real + ((1 - eps) * fake)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = autograd.Variable(self.Tensor(d_interpolates.shape).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(outputs=d_interpolates,
                                  inputs=interpolates,
                                  grad_outputs=fake,
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True,)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty


class Gradient_Difference_Loss(nn.Module):
    def __init__(self, alpha=1, chans=3, cuda=True):
        super(Gradient_Difference_Loss, self).__init__()
        self.alpha = alpha
        self.chans = chans
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        SobelX = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        SobelY = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        # SobelX = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        # SobelY = [[1, 2, -1], [0, 0, 0], [1, 2, -1]]
        self.Kx = Tensor(SobelX).expand(self.chans, 1, 3, 3)
        self.Ky = Tensor(SobelY).expand(self.chans, 1, 3, 3)

    def get_gradients(self, im):
        gx = F.conv2d(im, self.Kx, stride=1, padding=1, groups=self.chans)
        gy = F.conv2d(im, self.Ky, stride=1, padding=1, groups=self.chans)
        return gx, gy

    def forward(self, pred, true):
        # get graduent of pred and true
        gradX_true, gradY_true = self.get_gradients(true)
        grad_true = torch.abs(gradX_true) + torch.abs(gradY_true)
        gradX_pred, gradY_pred = self.get_gradients(pred)
        grad_pred_a = torch.abs(gradX_pred)**self.alpha + torch.abs(gradY_pred)**self.alpha
        # compute and return GDL
        return 0.5 * torch.mean(grad_true - grad_pred_a)



##新加判别器使用的损失
class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = torch.FloatTensor

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    # def forward(self, real_preds, fake_preds, target_is_real):
    #     if target_is_real:
    #         # Calculate R-a hinge loss for the discriminator
    #         r_f_diff = real_preds - torch.mean(fake_preds)
    #         f_r_diff = fake_preds - torch.mean(real_preds)
    #         loss = torch.mean(torch.nn.ReLU()(1 - r_f_diff)) + torch.mean(torch.nn.ReLU()(1 + f_r_diff))
    #     else:
    #         # Calculate R-a hinge loss for the generator生成器
    #         r_f_diff = real_preds - torch.mean(fake_preds)
    #         f_r_diff = fake_preds - torch.mean(real_preds)
    #         loss = torch.mean(torch.nn.ReLU()(1 + r_f_diff)) + torch.mean(torch.nn.ReLU()(1 - f_r_diff))
    #     return loss / 2
    def forward(self, real_preds, fake_preds):

        # Calculate R-a hinge loss for the discriminator
        r_f_diff = real_preds - torch.mean(fake_preds)
        f_r_diff = fake_preds - torch.mean(real_preds)
        loss = torch.mean(torch.nn.ReLU()(1 - r_f_diff)) + torch.mean(torch.nn.ReLU()(1 + f_r_diff))

        return loss / 2


#hzy总变分损失：
# 总变分函数：


def _tensor_size(t):
    return t.size()[1] * t.size()[2] * t.size()[3]


def tv_loss(x):

    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:, :, 1:, :])
    count_w = _tensor_size(x[:, :, :, 1:])
    h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
    w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
    return 2*(h_tv/count_h+w_tv/count_w)


class TV_Loss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TV_Loss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size=x.shape[0]
        return self.TVLoss_weight*tv_loss(x)/batch_size


class L2_L1_loss(nn.Module):
    def __init__(self,alpha=0.7):
        super(L2_L1_loss,self).__init__()
        self.alpha=alpha
        self.sL1_G = torch.nn.L1Loss()
        self.l2_G=torch.nn.MSELoss()

    def forward(self, real_preds, fake_preds, ):
        l1 = self.sL1_G(real_preds, fake_preds)
        l2 = self.l2_G(real_preds, fake_preds)
        loss =self.alpha * l2 + (1 - self.alpha) * l1
        return loss
    

class Grad_loss(nn.Module):

    def __init__(self):
        super(Grad_loss, self).__init__()
        self.L1_loss = torch.nn.L1Loss()
    def forward(self, pred, target):    
        pred_dx = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        target_dx = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        # 计算 y 方向梯度
        pred_dy = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        target_dy = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
        
        # 强迫预测的 T 的边缘，必须和 SAM 的 T 的边缘对齐
        loss = self.L1_loss(pred_dx, target_dx) + self.L1_loss(pred_dy, target_dy)
        return loss






#感知损失

# --- Perceptual loss network  --- #
class PerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg_model = vgg16(pretrained=True).features[:16]
        vgg_model = vgg_model.cuda()
        # vgg_model = nn.DataParallel(vgg_model, device_ids=device_ids)
        for param in vgg_model.parameters():
            param.requires_grad = False
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, pred_im, gt):
        loss = []
        pred_im_features = self.output_features(pred_im)
        gt_features = self.output_features(gt)
        for pred_im_feature, gt_feature in zip(pred_im_features, gt_features):
            loss.append(F.mse_loss(pred_im_feature, gt_feature))

        return sum(loss)/len(loss)

#SSIM loss

def gaussian_window(size, sigma):
    """
    生成高斯滤波器。
    """
    x = torch.arange(0, size, 1, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")).float()
    x -= (size - 1) / 2

    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()

    return g.view(1, 1, 1, -1).repeat(1, 1, size, 1)

class SSIM(nn.Module):
    def __init__(self,size=11, sigma=1.5, C1=0.01**2, C2=0.03**2):
        super(SSIM,self).__init__()
        self.size = size
        self.sigma = sigma
        self.C1 = C1
        self.C2 = C2


    def forward(self, img1, img2):
        window = gaussian_window(self.size, self.sigma).to(img1.device)  # window shape [1, 1, size, size]
        # 复制高斯滤波器以匹配输入图像的通道数
        window = window.repeat(img1.shape[1], 1, 1, 1)  # 现在 window shape [C, 1, size, size]

        mu1 = F.conv2d(img1, window, padding=self.size // 2, groups=img1.shape[1])
        mu2 = F.conv2d(img2, window, padding=self.size // 2, groups=img2.shape[1])

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1 ** 2, window, padding=self.size // 2, groups=img1.shape[1]) - mu1_sq
        sigma2_sq = F.conv2d(img2 ** 2, window, padding=self.size // 2, groups=img2.shape[1]) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.size // 2, groups=img1.shape[1]) - mu1_mu2

        value = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))
        return value.mean()












