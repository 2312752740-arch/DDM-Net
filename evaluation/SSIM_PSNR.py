import os
import numpy as np
from PIL import Image
from glob import glob
from os.path import join, basename
import pandas as pd

# 本地库
from imqual_utils import getSSIM, getPSNR

def SSIMs_PSNRs(gtr_dir, gen_dir, im_res=(256, 256)):
    """
    计算两组图片之间的SSIM和PSNR
    :param gtr_dir: 真实图片目录
    :param gen_dir: 生成图片目录
    :param im_res: 图片分辨率(默认256x256)
    :return: SSIM数组, PSNR数组
    """
    gtr_paths = sorted(glob(join(gtr_dir, "*.*")))
    gen_paths = sorted(glob(join(gen_dir, "*.*")))
    
    ssims, psnrs = [], []
    for gtr_path, gen_path in zip(gtr_paths, gen_paths):
        gtr_f = basename(gtr_path).split('.')[0]
        gen_f = basename(gen_path).split('.')[0]
        if (gtr_f == gen_f):
            # 打开并调整图片大小
            r_im = Image.open(gtr_path).resize(im_res)
            g_im = Image.open(gen_path).resize(im_res)
            
            # 计算RGB通道的SSIM
            ssim = getSSIM(np.array(r_im), np.array(g_im))
            ssims.append(ssim)
            
            # 计算L通道的PSNR
            r_im = r_im.convert("L")
            g_im = g_im.convert("L")
            psnr = getPSNR(np.array(r_im), np.array(g_im))
            psnrs.append(psnr)
    
    return np.array(ssims), np.array(psnrs)

def calculate_quality_for_all_models(gtr_dir, models_root_dir, output_csv=None):
    """
    计算所有模型文件夹的质量指标
    :param gtr_dir: GT图片目录
    :param models_root_dir: 包含多个模型结果的根目录
    :param output_csv: 结果保存的CSV文件路径
    """
    # 获取所有模型文件夹
    model_dirs = [d for d in os.listdir(models_root_dir) 
                 if os.path.isdir(os.path.join(models_root_dir, d))]
    
    results = []
    
    for model_dir in model_dirs:
        gen_dir = os.path.join(models_root_dir, model_dir)  # 假设结果在J文件夹下"J"
        
        if not os.path.exists(gen_dir):
            print(f"Warning: {gen_dir} does not exist, skipping")
            continue
        
        # 计算质量指标
        ssims, psnrs = SSIMs_PSNRs(gtr_dir, gen_dir)
        
        if len(ssims) == 0:
            print(f"Warning: No matching images found in {gen_dir}, skipping")
            continue
        
        # 收集结果
        result = {
            "Model": model_dir,
            "SSIM_Mean": np.mean(ssims),
            "SSIM_Std": np.std(ssims),
            "PSNR_Mean": np.mean(psnrs),
            "PSNR_Std": np.std(psnrs),
            "Num_Samples": len(ssims)
        }
        results.append(result)
        
        # 打印当前模型结果
        print(f"\nModel: {model_dir}")
        print(f"SSIM on {len(ssims)} samples - Mean: {result['SSIM_Mean']:.4f}, Std: {result['SSIM_Std']:.4f}")
        print(f"PSNR on {len(psnrs)} samples - Mean: {result['PSNR_Mean']:.2f}, Std: {result['PSNR_Std']:.2f}")
    
    # 将结果保存为DataFrame
    df = pd.DataFrame(results)
    
    # 按PSNR排序
    df = df.sort_values(by="PSNR_Mean", ascending=False)
    
    # 打印汇总结果
    print("\n=== Summary ===")
    print(df[["Model", "SSIM_Mean", "PSNR_Mean"]].to_string(index=False))
    
    # 保存到CSV
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\nResults saved to {output_csv}")

if __name__ == "__main__":
    # 配置路径
    GT_DIR = "./Dataset/test/testB"  # GT图片目录Dataset data_EUVP/test/testB
    MODELS_ROOT_DIR = "./output_ablation/ablation_a1/UIEB90/"  # 包含多个模型结果的目录 EUVP215 UIEB90
    OUTPUT_CSV = "./results/quality_metrics.csv"  # 结果保存路径
    
    # 计算所有模型的质量指标
    calculate_quality_for_all_models(
        gtr_dir=GT_DIR,
        models_root_dir=MODELS_ROOT_DIR,
        output_csv=OUTPUT_CSV
    )

# python evaluation/SSIM_PSNR.py