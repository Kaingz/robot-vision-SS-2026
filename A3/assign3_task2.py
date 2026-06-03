import numpy as np
import cv2

depths_paths = ["output_task1/depth_2011_10_03_drive_0047_sync_image_0000000005_image_03.npy", 
                "output_task1/depth_2011_10_03_drive_0047_sync_image_0000000086_image_03.npy",
                "output_task1/depth_2011_10_03_drive_0047_sync_image_0000000131_image_02.npy",
                "output_task1/depth_2011_10_03_drive_0047_sync_image_0000000209_image_02.npy",
                "output_task1/depth_2011_10_03_drive_0047_sync_image_0000000245_image_02.npy",
                "output_task1/depth_2011_10_03_drive_0047_sync_image_0000000296_image_03.npy",
                "output_task1/depth_2011_10_03_drive_0047_sync_image_0000000326_image_03.npy",
                "output_task1/depth_2011_10_03_drive_0047_sync_image_0000000395_image_02.npy",
                "output_task1/depth_2011_10_03_drive_0047_sync_image_0000000521_image_02.npy",
                "output_task1/depth_2011_10_03_drive_0047_sync_image_0000000734_image_03.npy"]

gt_paths = ["data_ass3/Task1_2/groundtruth/2011_10_03_drive_0047_sync_groundtruth_depth_0000000005_image_03.png", 
            "data_ass3/Task1_2/groundtruth/2011_10_03_drive_0047_sync_groundtruth_depth_0000000086_image_03.png",
            "data_ass3/Task1_2/groundtruth/2011_10_03_drive_0047_sync_groundtruth_depth_0000000131_image_02.png",
            "data_ass3/Task1_2/groundtruth/2011_10_03_drive_0047_sync_groundtruth_depth_0000000209_image_02.png",
            "data_ass3/Task1_2/groundtruth/2011_10_03_drive_0047_sync_groundtruth_depth_0000000245_image_02.png",
            "data_ass3/Task1_2/groundtruth/2011_10_03_drive_0047_sync_groundtruth_depth_0000000296_image_03.png",
            "data_ass3/Task1_2/groundtruth/2011_10_03_drive_0047_sync_groundtruth_depth_0000000326_image_03.png",
            "data_ass3/Task1_2/groundtruth/2011_10_03_drive_0047_sync_groundtruth_depth_0000000395_image_02.png",
            "data_ass3/Task1_2/groundtruth/2011_10_03_drive_0047_sync_groundtruth_depth_0000000521_image_02.png",
            "data_ass3/Task1_2/groundtruth/2011_10_03_drive_0047_sync_groundtruth_depth_0000000734_image_03.png"] 

print("\n")
print("i |                                     File                                    | RMSE")
print("--+-----------------------------------------------------------------------------+--------------------")

for i in range(len(depths_paths)):
    depth_pred = np.load(depths_paths[i])

    gt = cv2.imread(gt_paths[i], cv2.IMREAD_ANYDEPTH)

    gt = gt.astype(float) / 256.0

    mask = (gt > 0) & (gt < 120.0)

    gt = gt[mask]

    depth_pred = depth_pred[mask]

    rmse = np.sqrt(np.mean((gt - depth_pred) ** 2))

    print(str(i) + " | " + depths_paths[i] + " | " +  str(rmse) + "m")

print("\n")