import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm

def get_udcp(image_float, patch_size=15):
    b, g, r = cv2.split(image_float)
    min_channel = np.minimum(np.minimum(b, g), r * 0.5 + 0.5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_channel, kernel)
    num_pixels = dark_channel.size
    num_brightest = int(max(num_pixels * 0.001, 1))
    indices = np.argpartition(dark_channel.flatten(), -num_brightest)[-num_brightest:]
    flat_img = image_float.reshape(-1, 3)
    A_est = np.max(flat_img[indices], axis=0)
    A_max = np.max(A_est)
    t = 1 - 0.95 * (dark_channel / (A_max + 1e-6))
    return np.clip(t, 0.1, 1.0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_paths = sorted([os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
                          if os.path.splitext(f)[1].lower() in image_extensions])
    print(f"Found {len(image_paths)} images to process.")
    means = []
    for img_path in tqdm(image_paths, desc="UDCP-only"):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(args.output_dir, f"{base_name}.jpg")
        if os.path.exists(output_path):
            continue
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"Warning: cannot read {img_path}, skipped.")
            continue
        img_float = img_bgr.astype(np.float32) / 255.0
        t_initial = get_udcp(img_float)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        t_final = cv2.ximgproc.guidedFilter(guide=gray, src=t_initial.astype(np.float32), radius=60, eps=1e-6)
        cv2.imwrite(output_path, (np.clip(t_final, 0, 1) * 255).astype(np.uint8))
        means.append(float(t_final.mean()))
    if means:
        print(f"t-map stats: mean={np.mean(means):.4f} min={np.min(means):.4f} max={np.max(means):.4f}")
    print("Processing complete.")

if __name__ == '__main__':
    main()
