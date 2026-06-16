import torch
from unidepth.models import UniDepthV1
from PIL import Image
import numpy as np
import cv2

model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14")

device = torch.device("cpu")
model = model.to(device)

base_path = "data_ass3/Task1_2/images/"
image_paths = ["2011_10_03_drive_0047_sync_image_0000000005_image_03", 
               "2011_10_03_drive_0047_sync_image_0000000086_image_03",
               "2011_10_03_drive_0047_sync_image_0000000131_image_02",
               "2011_10_03_drive_0047_sync_image_0000000209_image_02",
               "2011_10_03_drive_0047_sync_image_0000000245_image_02",
               "2011_10_03_drive_0047_sync_image_0000000296_image_03",
               "2011_10_03_drive_0047_sync_image_0000000326_image_03",
               "2011_10_03_drive_0047_sync_image_0000000395_image_02",
               "2011_10_03_drive_0047_sync_image_0000000521_image_02",
               "2011_10_03_drive_0047_sync_image_0000000734_image_03"] 

for image_path in image_paths:

    # Open image and convert to tensor, then feed into network
    image = torch.from_numpy(np.array(Image.open(base_path + image_path + ".png").convert("RGB"))).permute(2, 0, 1).to(device)

    predictions = model.infer(image)

    # Extract depth and save
    depth = predictions["depth"].squeeze().cpu().numpy()
    np.save("output_task1/depth_" + image_path, depth)

    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_visual = depth_normalized.astype(np.uint8)

    depth_colored = cv2.applyColorMap(depth_visual, cv2.COLORMAP_INFERNO)

    cv2.imwrite("output_task1/depth_" + image_path + ".png", depth_colored)
