import os
import glob
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


 #root=./Dataset/
# dataset_name=UIEB
# class Dataloader(Dataset):
#     def __init__(self, root, dataset_name, transforms_=None):
#         self.transform = transforms.Compose(transforms_)
#         print(root, dataset_name)
#         self.filesA, self.filesB = self.get_file_paths(root, dataset_name)
#         self.len = min(len(self.filesA), len(self.filesB))

#     def __getitem__(self, index):
#         img_A = Image.open(self.filesA[index % self.len])
#         img_B = Image.open(self.filesB[index % self.len])
#         if np.random.random() < 0.5:
#             img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
#             img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")
#         img_A = self.transform(img_A)
#         img_B = self.transform(img_B)
#         return {"A": img_A, "B": img_B}

#     def __len__(self):
#         return self.len

#     def get_file_paths(self, root, dataset_name):
#         if dataset_name=='UIEB':
#             filesA, filesB = [], []
#             sub_dirs = ['train']
#             for sd in sub_dirs:
#                 filesA += sorted(glob.glob(os.path.join(root, sd, 'trainA') + "/*.*"))
#                 filesB += sorted(glob.glob(os.path.join(root, sd, 'trainB') + "/*.*"))
#         return filesA, filesB 


#\Dataset\train\trainB
class Dataloader(Dataset):
    def __init__(self, root, dataset_name, transforms_=None):
        # --- 核心改动 1: 智能创建 T-map 的 transform ---
        # 假设传入的 transforms_ 是一个列表，且 Normalize 是最后一项
        self.transform = transforms.Compose(transforms_)
        # 复用列表，创建不含 Normalize 的版本给 T-map 使用
        self.t_map_transform = transforms.Compose(transforms_[:-1])  # <--- 最小改动：智能复用
        # --- 结束 ---

        print(root, dataset_name)

        # --- 核心改动 2: 接收第三个文件列表 ---
        self.filesA, self.filesB, self.filesT = self.get_file_paths(root, dataset_name)  # <--- 接收 T
        # --- 结束 ---

        # --- 核心改动 3: 更新数据集长度 ---
        self.len = min(len(self.filesA), len(self.filesB), len(self.filesT))  # <--- 包含 T
        # --- 结束 ---

    def __getitem__(self, index):
        # --- 核心改动 4: 加载并转换 T-map ---
        img_A = Image.open(self.filesA[index % self.len]).convert("RGB")
        img_B = Image.open(self.filesB[index % self.len]).convert("RGB")
        img_T = Image.open(self.filesT[index % self.len]).convert("L")  # <--- 加载 T (L=灰度图)

        if np.random.random() < 0.5:
            # 使用 PIL 的 transpose 方法，更简洁
            img_A = img_A.transpose(Image.FLIP_LEFT_RIGHT)
            img_B = img_B.transpose(Image.FLIP_LEFT_RIGHT)
            img_T = img_T.transpose(Image.FLIP_LEFT_RIGHT)  # <--- 同步翻转 T

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)
        img_T = self.t_map_transform(img_T)  # <--- 使用为 T 准备的 transform

        return {"A": img_A, "B": img_B, "T": img_T}  # <--- 返回 T
        # --- 结束 ---

    def __len__(self):
        return self.len

    def get_file_paths(self, root, dataset_name):
        # --- 核心改动 5: 增加寻找 T 文件的逻辑 ---
        if dataset_name == 'UIEB':
            filesA, filesB, filesT = [], [], []  # <--- 初始化 T 列表
            sub_dirs = ['train']
            for sd in sub_dirs:
                # 假设 'T' 文件夹名为 'trainT'，与 'trainA', 'trainB' 并列
                pathT = os.path.join(root, sd, 'trainT')  # <--- 定义 T 文件夹路径

                filesA += sorted(glob.glob(os.path.join(root, sd, 'trainA') + "/*.*"))
                filesB += sorted(glob.glob(os.path.join(root, sd, 'trainB') + "/*.*"))
                filesT += sorted(glob.glob(pathT + "/*.*"))  # <--- 寻找 T 文件

            return filesA, filesB, filesT  # <--- 返回三个列表

        # 为您的另一个数据集也增加逻辑
        elif dataset_name == 'A_dataset2000':
            filesA, filesB, filesT = [], [], []
            sub_dirs = ['train']
            for sd in sub_dirs:
                pathA = os.path.join(root, dataset_name, sd, 'A')
                pathB = os.path.join(root, dataset_name, sd, 'B')
                pathT = os.path.join(root, dataset_name, sd, 'T')  # T 文件夹的路径

                filesA += sorted(glob.glob(pathA + "/*.*"))
                filesB += sorted(glob.glob(pathB + "/*.*"))
                filesT += sorted(glob.glob(pathT + "/*.*"))

            return filesA, filesB, filesT

        # 如果还有其他数据集，可以在此添加
        else:
            raise NotImplementedError(f"Dataset '{dataset_name}' loading logic is not implemented.")
