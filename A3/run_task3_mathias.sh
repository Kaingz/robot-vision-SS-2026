#!/bin/bash

find output_task3 -type f -delete

source /home/mathias/RV/robot-vision-SS-2026/A1/unimatch/env/bin/activate

python3 unimatch/main_stereo.py \
    --inference_dir_left data_ass3/Task3/rectified_images/image_2 \
    --inference_dir_right data_ass3/Task3/rectified_images/image_3 \
    --output_path output_task3/unimatch_output \
    --inference_size 384 1248 \
    --num_scales 2 \
    --reg_refine \
    --num_reg_refine 3 \
    --upsample_factor 4 \
    --attn_type self_swin2d_cross_swin1d \
    --attn_splits_list 2 8 \
    --corr_radius_list -1 4 \
    --prop_radius_list -1 1 \
    --resume pretrained/gmstereo-scale2-regrefine3-resumeflowthings-kitti15-04487ebf.pth \
    --padding_factor 32 \
    --save_pfm_disp \

deactivate

source /home/mathias/RV/envs/Unidepth/bin/activate

python3 assign3_task3.py