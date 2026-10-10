
"""
# > Script for measuring quantitative performances in terms of
#    - Structural Similarity Metric (SSIM)
#    - Peak Signal to Noise Ratio (PSNR)
# > Maintainer: https://github.com/xahidbuffon
"""
import os

## python libs
import numpy as np
from PIL import Image
from glob import glob
from os.path import join
from ntpath import basename
## local libs
from imqual_utils import getSSIM, getPSNR
import csv
from tqdm import tqdm




## compares avg ssim and psnr
def calculate_metrics_for_images(gtr_dir, gen_dir, im_res=(256, 256), output_csv="image_metrics.csv"):
    """
    计算每张图片的SSIM和PSNR，并保存到CSV文件
    参数:
        gtr_dir: 真实图像目录
        gen_dir: 生成图像目录
        im_res: 图像调整尺寸 (默认256x256)
        output_csv: 输出CSV文件名
    """
    # 获取图像路径并排序确保匹配
    gtr_paths = sorted(glob(join(gtr_dir, "*.*")))
    gen_paths = sorted(glob(join(gen_dir, "*.*")))

    results = []

    # 使用tqdm显示进度条
    for gtr_path, gen_path in tqdm(zip(gtr_paths, gen_paths), total=len(gtr_paths), desc="计算指标"):
        gtr_f = basename(gtr_path).split('.')[0]
        gen_f = basename(gen_path).split('.')[0]

        if gtr_f == gen_f:  # 确保文件名匹配
            try:
                # 打开并调整图像大小
                r_im = Image.open(gtr_path).resize(im_res)
                g_im = Image.open(gen_path).resize(im_res)

                # 计算SSIM (RGB通道)
                ssim = getSSIM(np.array(r_im), np.array(g_im))

                # 计算PSNR (L通道/灰度)
                r_gray = r_im.convert("L")
                g_gray = g_im.convert("L")
                psnr = getPSNR(np.array(r_gray), np.array(g_gray))

                results.append({
                    'Image': basename(gtr_path),
                    'SSIM': f"{ssim:.3f}",  # 保留3位小数
                    'PSNR': f"{psnr:.3f}"  # 保留3位小数
                })

            except Exception as e:
                print(f"处理 {basename(gtr_path)} 时出错: {str(e)}")
                results.append({
                    'Image': basename(gtr_path),
                    'SSIM': "Error",
                    'PSNR': "Error"
                })

    # 保存结果到CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Image', 'SSIM', 'PSNR']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n结果已保存到 {output_csv}")


# 使用示例
if __name__ == "__main__":
    # 设置你的路径Q_re324D:\MS-hu\A_daughter\compare_dates\six_datasets\A_test_UIEB
    gtr_dir = "D:/MS-hu/A_daughter/compare_dates/six_datasets/A_test_UIEB/A_ref"  # 真实图像目录
    gen_dir ="D:/MS-hu/A_daughter/compare_dates/six_datasets/A_test_UIEB/Q_re324"  # 生成图像目录

    # 计算并保存指标
    calculate_metrics_for_images(gtr_dir, gen_dir, output_csv="image_quality_metrics.csv")

