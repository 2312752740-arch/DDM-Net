import torch
import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# --- 1. SAM模型加载 ---
def load_sam_model(model_type, checkpoint_path, device):
    """加载SAM模型并返回一个自动掩码生成器"""
    print(f"Loading SAM model: {model_type} from {checkpoint_path}...")
    try:
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )
        print("SAM model loaded successfully.")
        return mask_generator
    except Exception as e:
        print(f"Error loading SAM model: {e}")
        print("Please ensure the model type and checkpoint path are correct.")
        exit()

# --- 2. 初步传输图估算 (UDCP) ---
# def get_udcp(image_float, patch_size=15, omega=0.95):
#     """计算水下暗通道先验(UDCP)来估计初始传输图"""
#     b, g, r = cv2.split(image_float)
#     dark_channel = np.minimum(b, g)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
#     dark_channel_min = cv2.erode(dark_channel, kernel)
    
#     A_c = np.max([np.max(b), np.max(g)]) # 简化的背景光估计
    
#     image_norm = image_float / A_c
#     dark_channel_norm = np.minimum(image_norm[:,:,0], image_norm[:,:,1])
#     dark_channel_norm_min = cv2.erode(dark_channel_norm, kernel)
    
#     t_initial = 1 - omega * dark_channel_norm_min
#     return np.clip(t_initial, 0.0, 1.0)
def get_udcp(image_float, patch_size=15):
    b, g, r = cv2.split(image_float)
    
    # 这里的关键是：水下红通道虽然衰减快，但不能直接忽略，
    # 否则红色物体会被误认为是远处的背景（传输率低）。
    # 简单的改进：用 1-R 来反转红通道参与计算
    
    # 方案 A: 经典的 DCP (适用于去雾，水下不一定好)
    # min_channel = np.minimum(np.minimum(b, g), r)
    
    # 方案 B: UDCP (你现在的)
    # min_channel = np.minimum(b, g) 
    
    # 方案 C (推荐): 修正的 UDCP
    # 给予红通道一个较小的权重参与比较，防止红色物体丢失
    min_channel = np.minimum(np.minimum(b, g), r * 0.5 + 0.5) 

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_channel, kernel)
    
    # 估计 A (取前 0.1% 最亮的点)
    num_pixels = dark_channel.size
    num_brightest = int(max(num_pixels * 0.001, 1))
    indices = np.argpartition(dark_channel.flatten(), -num_brightest)[-num_brightest:]
    
    # 在原图中找对应的最大亮度作为 A
    flat_img = image_float.reshape(-1, 3)
    A_est = np.max(flat_img[indices], axis=0)
    A_max = np.max(A_est) # 或者分别处理通道
    
    # 计算 T
    t = 1 - 0.95 * (dark_channel / (A_max + 1e-6))
    return np.clip(t, 0.1, 1.0) # 设置下限，防止除0

# --- 3. SAM掩码精炼 ---
def refine_t_with_sam(t_initial, sam_masks):
    """利用SAM生成的掩码来平滑和精炼传输图"""
    if not sam_masks:
        return t_initial
        
    t_refined = np.zeros_like(t_initial, dtype=np.float32)
    sorted_masks = sorted(sam_masks, key=(lambda x: x['area']), reverse=True)
    
    for mask_info in sorted_masks:
        mask = mask_info['segmentation']
        t_median_in_mask = np.median(t_initial[mask])
        t_refined[mask] = t_median_in_mask
        
    return t_refined

# --- 4. 主流程函数 ---
def generate_and_save_transmission(image_path, output_path, mask_generator):
    """
    对单张图片执行完整流程并保存结果
    """
    try:
        # 1. 加载图像
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            return
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_float = img_bgr.astype(np.float32) / 255.0

        # 2. 初步估算 (UDCP)
        t_initial_udcp = get_udcp(img_float)

        # 3. SAM 分割
        sam_masks = mask_generator.generate(img_rgb)

        # 4. SAM 精炼
        t_refined_sam = refine_t_with_sam(t_initial_udcp, sam_masks)

        # 5. 引导滤波进行最终优化
        gray_img_for_guide = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        radius = 60
        eps = 1e-6
        t_final = cv2.ximgproc.guidedFilter(
            guide=gray_img_for_guide,
            src=t_refined_sam.astype(np.float32),
            radius=radius,
            eps=eps
        )

        # 6. 保存传输图
        # 归一化到 0-255 并转换为8位无符号整型
        t_final_normalized = (np.clip(t_final, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(output_path, t_final_normalized)

    except Exception as e:
        print(f"An error occurred while processing {image_path}: {e}")


# --- 5. 命令行接口与批量处理 ---
def main():
    parser = argparse.ArgumentParser(description="Generate high-quality transmission maps for underwater images.")
    parser.add_argument('--input_dir', default="/home/xyl/hzy/seg/Dataset/test/testA", help='Path to the directory containing input images.')
    parser.add_argument('--output_dir', default="Dataset/train1/trainT", help='Path to the directory where transmission maps will be saved.')
    parser.add_argument('--sam_model_type', type=str, default='vit_b', help='SAM model type (e.g., vit_h, vit_l, vit_b).')
    parser.add_argument('--sam_checkpoint', default="model/weight/sam_vit_b_01ec64.pth", help='Path to the SAM model checkpoint file (.pth).')
    
    args = parser.parse_args()

    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型
    mask_generator = load_sam_model(args.sam_model_type, args.sam_checkpoint, device)

    # 查找所有图片
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_paths = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if os.path.splitext(f)[1].lower() in image_extensions]

    # 批量处理
    print(f"Found {len(image_paths)} images to process.")
    for img_path in tqdm(image_paths, desc="Processing images"):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(args.output_dir, f"{base_name}.jpg")
        
        # 如果文件已存在则跳过，方便中断后继续
        if os.path.exists(output_path):
            # print(f"Skipping {img_path}, output already exists.")
            continue

        generate_and_save_transmission(img_path, output_path, mask_generator)

    print("Processing complete.")
    print(f"Transmission maps saved to: {args.output_dir}")


if __name__ == '__main__':
    main()