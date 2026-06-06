import torch
import matplotlib.pyplot as plt
from unidepth.models import UniDepthV1
from unimatch.utils.file_io import read_disp
from PIL import Image
import numpy as np
import cv2
import os

# Run Unidepth
model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14")

device = torch.device("cpu")
model = model.to(device)

base_path = "data_ass3/Task3/rectified_images/image_2/"

unidepth_output_path = "output_task3/unidepth_output/"
unimatch_output_path = "output_task3/unimatch_output/"
gt_path = "data_ass3/Task3/GT_disparities/disp_noc_0"

for image_path in sorted(os.listdir(base_path)):
    image = torch.from_numpy(np.array(Image.open(base_path + image_path).convert("RGB"))).permute(2, 0, 1).to(device)

    predictions = model.infer(image)

    depth = predictions["depth"].squeeze().cpu().numpy()
    np.save("output_task3/unidepth_output/" + image_path[:-4], depth)

    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_visual = depth_normalized.astype(np.uint8)

    depth_colored = cv2.applyColorMap(depth_visual, cv2.COLORMAP_INFERNO)

    cv2.imwrite("output_task3/unidepth_output/" + image_path, depth_colored)

# Construct Depth Images

unidepth_depth_images = []
unidepth_masks = []
unimatch_depth_images = []
unimatch_masks = []
gt_depth_images = []
gt_masks = []

f = 721
B = 0.54

filenames = []

for filename in sorted(os.listdir(unidepth_output_path)):
    if filename.endswith('.npy'):
        filenames.append(filename[:-4])
        depth = np.load(os.path.join(unidepth_output_path, filename))
        unidepth_depth_images.append(depth)
        mask = (depth > 0.0) & (depth < 120.0)
        unidepth_masks.append(mask)

for filename in sorted(os.listdir(unimatch_output_path)):
    if filename.endswith('.pfm'):
        disp = read_disp(os.path.join(unimatch_output_path, filename))
        depth = np.where(disp > 0, f * B / disp, 0)
        mask = (depth > 0.0) & (depth < 120.0)
        unimatch_masks.append(mask)
        unimatch_depth_images.append(depth)

for filename in sorted(os.listdir(gt_path)):
    gt = cv2.imread(os.path.join(gt_path, filename), cv2.IMREAD_ANYDEPTH).astype(np.float32)
    gt = gt / 256.0
    gt = np.where(gt > 0, f * B / gt, 0)
    mask = (gt > 0.0) & (gt < 120.0)
    gt_masks.append(mask)

    gt_depth_images.append(gt)

diff_imgs_unimatch = []
gray_imgs_unimatch = []
diff_imgs_unidepth = []
gray_imgs_unidepth = []

# Analysis
rmse_values = np.zeros(len(gt_depth_images))
print("i | RMSE")
print("--+--------------------")
for i in range(0, len(gt_depth_images)):
    # Compute RMSE
    mask = unimatch_masks[i] & gt_masks[i]
    rmse_values[i] = np.sqrt(np.mean((gt_depth_images[i][mask] - unimatch_depth_images[i][mask])**2))
    print(str(i) + " | " +  str(rmse_values[i]) + "m")
    # Compute Diffs
    diff_imgs_unimatch.append(np.zeros_like(gt_depth_images[i], dtype=np.float32))
    diff_imgs_unimatch[i][mask] = np.abs(gt_depth_images[i][mask] - unimatch_depth_images[i][mask])

    mask_d = unidepth_masks[i] & gt_masks[i]
    diff_imgs_unidepth.append(np.zeros_like(gt_depth_images[i], dtype=np.float32))
    diff_imgs_unidepth[i][mask_d] = np.abs(gt_depth_images[i][mask_d] - unidepth_depth_images[i][mask_d])

    max_value = max(diff_imgs_unimatch[i].max(), diff_imgs_unidepth[i].max())
    gray_imgs_unimatch.append(np.zeros_like(gt_depth_images[i], dtype=np.uint8))
    gray_imgs_unimatch[i][mask] = (diff_imgs_unimatch[i][mask] / max_value * 255).astype(np.uint8)
    cv2.imwrite(f"output_task3/grayscale_unimatch_diff/{filenames[i]}.png", gray_imgs_unimatch[i])

    gray_imgs_unidepth.append(np.zeros_like(gt_depth_images[i], dtype=np.uint8))
    gray_imgs_unidepth[i][mask_d] = (diff_imgs_unidepth[i][mask_d] / max_value * 255).astype(np.uint8)
    cv2.imwrite(f"output_task3/grayscale_unidepth_diff/{filenames[i]}.png", gray_imgs_unidepth[i])

    # Histograms
    plt.figure()
    plt.hist(diff_imgs_unimatch[i][mask], bins=100, range=(0,10))
    plt.title('Histogram Unimatch Depth Errors')
    plt.xlabel('Absolute difference in depth [m]')
    plt.ylabel('Number of Occurences')
    plt.tight_layout()
    plt.savefig(f'output_task3/histograms_unimatch/hist_{filenames[i]}.svg')
    plt.close()

    plt.figure()
    plt.hist(diff_imgs_unidepth[i][mask_d], bins=100, range=(0,10))
    plt.title('Histogram Unidepth Depth Errors')
    plt.xlabel('Absolute difference in depth [m]')
    plt.ylabel('Number of Occurences')
    plt.tight_layout()
    plt.savefig(f'output_task3/histograms_unidepth/hist_{filenames[i]}.svg')
    plt.close()


np.save(os.path.join("output_task3/", "rmse_stereo_gt"), rmse_values)