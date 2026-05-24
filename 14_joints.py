import os
import sys
import cv2
import traceback
import numpy as np
from DataReader.read import DataReader
from preprocessing import get_crop_box, render_depth, MARGIN, MIN_PIXELS


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
SAMPLES_DIR=os.path.join(ROOT_DIR, 'Samples')
# SAMPLES_DIR = os.path.join(ROOT_DIR, 'cloth3d++_subset', 'cloth3d++_subset')
OUTPUT_DIR=os.path.join(ROOT_DIR, 'data')
SAMPLES_DIR=os.path.join(ROOT_DIR, 'Samples')

reader = DataReader()
SMPL_14_JOINTS= [
    1,#left hip
    2,#right hip
    4,#left knee
    5,#right knee
    7,#left ankle
    8,#right ankle
    12,#neck
    15,#head
    16,#left shoulder
    17,#right shoulder
    18,#left elbow
    19,#right elbow
    20,#left wrist
    21 #right wrist
]

def project_smpl_joints_14(sample, frame, crop_box=None, image_shape=(480, 640)):
    """
    Projects 14 SMPL joints to image coordinates.
    Returns:
        joints_2d:[14,2] with x, y coordinates
        joints_valid:[14] boolean mask
    """
    H,W=image_shape
    #Get gender so we know which SMPL model was used
    gender, _, _, _=reader.read_smpl_params(sample,frame)
    #Human vertices in the same coordinate system used by render_depth
    V,_=reader.read_human(sample, frame)
    #Regress 24 SMPL joints from the posed human mesh
    J_3d=reader.smpl[gender].J_regressor.dot(V)
    #select 14 main joints
    J_3d=J_3d[SMPL_14_JOINTS]
    #Project 3D joints to image
    P=reader.read_camera(sample)
    ones=np.ones((J_3d.shape[0], 1), dtype=np.float32)
    J_h=np.hstack([J_3d, ones])
    uvw=(P @ J_h.T).T
    x=uvw[:, 0]/uvw[:, 2]
    y=uvw[:, 1]/uvw[:, 2]
    z=uvw[:, 2]
    joints_2d=np.stack([x, y], axis=1).astype(np.float32)
    #valid in original full image
    joints_valid = ((z > 0) & (joints_2d[:, 0] >= 0) & (joints_2d[:, 0] < W) & (joints_2d[:, 1] >= 0) & (joints_2d[:, 1] < H))
    #Convert from full-image coordinates to cropped-image coordinates
    if crop_box is not None:
        x1, y1, x2, y2=crop_box
        joints_2d[:, 0]-=x1
        joints_2d[:, 1]-=y1
        crop_w=x2 - x1
        crop_h=y2 - y1
        joints_valid= (joints_valid&(joints_2d[:, 0]>=0)&(joints_2d[:, 0]<crop_w) & (joints_2d[:, 1]>=0) & (joints_2d[:, 1]<crop_h))
    return joints_2d, joints_valid


if __name__ == "__main__":
    img_dir=os.path.join(OUTPUT_DIR,'image')
    depth_dir=os.path.join(OUTPUT_DIR,'depth')
    os.makedirs(img_dir,exist_ok=True)
    os.makedirs(depth_dir,exist_ok=True)
    joints_dir = os.path.join(OUTPUT_DIR, "joints")
    os.makedirs(joints_dir, exist_ok=True)
    saved=0
    skipped=0
    for sample in sorted(os.listdir(SAMPLES_DIR)):
        frames_dir=os.path.join(SAMPLES_DIR, sample, 'frames')
        if not os.path.isdir(frames_dir):
            continue
        for fname in sorted(os.listdir(frames_dir)):
            if not fname.endswith('.png'):
                continue
            frame_num=os.path.splitext(fname)[0]       
            frame_idx=int(frame_num) - 1               
            base_name=f"{sample}_{frame_num}"
            rgba=cv2.imread(os.path.join(frames_dir, fname), cv2.IMREAD_UNCHANGED)
            if rgba is None or rgba.shape[2] != 4:
                skipped+=1
                continue
            rgb=rgba[:, :, :3]
            mask=rgba[:, :, 3]>0
            H,W=rgb.shape[:2]
            box=get_crop_box(mask, MARGIN, MIN_PIXELS)
            if box is None:
                print(f"{base_name} out of scene")
                skipped += 1
                continue
            x1,y1,x2,y2=box
            if x1<0 or y1<0 or x2>W or y2>H:
                print(f"{base_name} crop out of bounds")
                skipped += 1
                continue
            joints_2d, joints_valid = project_smpl_joints_14(sample,frame_idx,crop_box=(x1, y1, x2, y2),image_shape=(H, W))
            joints_data = np.concatenate([joints_2d, joints_valid[:, None].astype(np.float32)],axis=1)
            np.save(os.path.join(joints_dir, base_name + ".npy"), joints_data)
            saved+=1
            if saved%50==0:
                print(f"Saved {saved} frames")

    print(f"Saved {saved} and skipped {skipped}")