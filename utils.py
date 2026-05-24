import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
from pathlib import Path

class DataGenerator(Dataset):
    def __init__(self, data_list, root, target_size=(256, 256), max_depth=None, data_aug=None,  loss_method=None):
        super().__init__()
        'Initialization'
        self.root = root
        self.target_size= target_size
        self.max_depth= max_depth
        if self.max_depth is None:
            raise ValueError("max_depth must be provided")
        self.data_list= [f for f in data_list if "_aug" not in f]
        image_dir = os.path.join(self.root, "image")
        if data_aug is None or data_aug == 0:
            pass
        elif data_aug == 1:
            aug_1= [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(".jpg") and "_aug1" in f]
            self.data_list = self.data_list + aug_1
        elif data_aug == 2:
            aug_2= [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(".jpg") and "_aug2" in f]
            self.data_list = self.data_list + aug_2
        else:
            raise ValueError("data_aug must be None, 0, 1 or 2")
        self.loss_method=loss_method

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        f= self.data_list[index]
        h,w= self.target_size
        depth_path= os.path.join(self.root, "depth", f + ".npy")
        dpt= np.load(depth_path).astype(np.float32)
        original_mask=(dpt > 0).astype(np.uint8)
        dpt= cv2.resize(dpt, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(original_mask,(w, h),interpolation=cv2.INTER_NEAREST).astype(bool)
        dpt[~mask]= 0.0
        
        #Global normalization
        dpt_global=np.zeros_like(dpt, dtype=np.float32)
        if np.any(mask):
            dpt_global[mask]= dpt[mask]/(self.max_depth+1e-8)
            dpt_global=np.clip(dpt_global,0.0,1.0)
        #Per image normalization
        dpt_per_image=np.zeros_like(dpt, dtype=np.float32)
        if np.any(mask):
            min_val= dpt[mask].min()
            max_val= dpt[mask].max()
            dpt_per_image[mask]= (dpt[mask]-min_val)/(max_val-min_val+1e-8)
        
        dpt_global= np.expand_dims(dpt_global, axis=0)
        dpt_per_image= np.expand_dims(dpt_per_image, axis=0)
        #Load image
        img_path= os.path.join(self.root, "image", f + ".jpg")
        img=cv2.imread(img_path)
        img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img= cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        img = img / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))

        img = torch.from_numpy(img).float()
        dpt_global = torch.from_numpy(dpt_global).float()
        dpt_per_image = torch.from_numpy(dpt_per_image).float()

        sample = {"global": dpt_global,"per_image": dpt_per_image}
        if self.loss_method == "multitask" or self.loss_method=="pose_guided":
            joints_path = os.path.join(self.root, "joints", f + ".npy")
            joints_data = np.load(joints_path).astype(np.float32)
            joints_xy = joints_data[:, :2]
            joints_valid = joints_data[:, 2]
            orig_h, orig_w = original_mask.shape
            scale_x = w / orig_w
            scale_y = h / orig_h
            joints_xy[:, 0] = joints_xy[:, 0] * scale_x
            joints_xy[:, 1] = joints_xy[:, 1] * scale_y
            sample["joints_xy"] = torch.from_numpy(joints_xy).float()
            sample["joints_valid"] = torch.from_numpy(joints_valid).float()
        return img, sample

    

def read_data(DATA):
    with open(os.path.join(DATA, 'train.txt'), 'r') as f:
        train_list = f.readlines()
        for i in range(len(train_list)):
            train_list[i] = train_list[i].rsplit('\n',1)[0]
    with open(os.path.join(DATA, 'validation.txt'), 'r') as f:
        validation_list = f.readlines()
        for i in range(len(validation_list)):
            validation_list[i] = validation_list[i].rsplit('\n',1)[0]
    with open(os.path.join(DATA, 'test.txt'), 'r') as f:
        test_list = f.readlines()
        for i in range(len(test_list)):
            test_list[i] = test_list[i].rsplit('\n',1)[0]
    return train_list, validation_list, test_list


def generate_train_val_test_txt():
    frames_dir= Path("data/image")
    output_dir= frames_dir.parent
    train_file= output_dir/ "train.txt"
    val_file= output_dir/ "validation.txt"
    test_file= output_dir/ "test.txt"
    train_frames= []
    val_frames= []
    test_frames= []
    for img_path in sorted(frames_dir.glob("*.jpg")):
        filename= img_path.name
        video_name= filename.split("_")[0]
        try:
            video_id= int(video_name)
        except ValueError:
            print(f"Skipping file with unexpected name: {filename}")
            continue
        if 1 <= video_id <= 152:
            train_frames.append(filename.split(".")[0])
        elif 153 <= video_id <= 168:
            val_frames.append(filename.split(".")[0])
        else:
            test_frames.append(filename.split(".")[0])
    train_file.write_text("\n".join(train_frames))
    val_file.write_text("\n".join(val_frames))
    test_file.write_text("\n".join(test_frames))
    print(f"Train frames: {len(train_frames)}")
    print(f"Validation frames: {len(val_frames)}")
    print(f"Test frames: {len(test_frames)}")

def compute_train_max_depth(data_list, root):
    max_depth= 0.0
    for f in data_list:
        depth_path= os.path.join(root, "depth", f + ".npy")
        dpt= np.load(depth_path).astype(np.float32)
        mask= dpt > 0
        if np.any(mask):
            max_depth= max(max_depth,float(dpt[mask].max()))
    return max_depth

if __name__ == "__main__":
    generate_train_val_test_txt()