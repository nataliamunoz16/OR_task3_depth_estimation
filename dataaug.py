import numpy as np
import cv2
import os
from pathlib import Path
import random
import copy

#temp
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.patches as patches
from utils import read_data


def clear_background_noise(item):
    if item<5:
        return 0
    else:
        return item

def get_bbox(img, margin=0):
    mask= (img[:, :, 0] != 0) | (img[:, :, 1] != 0) | (img[:, :, 2] != 0)
    ys,xs= np.where(mask)
    if len(xs)== 0 or len(ys)== 0:
        return None
    H, W = img.shape[:2]
    xmin= max(xs.min()- margin, 0)
    xmax= min(xs.max()+ margin, W - 1)
    ymin= max(ys.min()- margin, 0)
    ymax= min(ys.max()+ margin, H - 1)
    return xmin,xmax,ymin,ymax

def image_shift(image, depth, xmin, xmax, ymin, ymax):
    H,W,_ = image.shape
    newimage = np.zeros_like(image)
    newdepth = np.zeros_like(depth)
    shift_left= xmin
    shift_right= W - xmax - 1
    shift_up= ymin
    shift_down= H - ymax - 1
    shift_y = random.randint(-shift_up,shift_down)
    shift_x = random.randint(-shift_left,shift_right)
    src_y0= ymin
    src_y1= ymax + 1
    src_x0= xmin
    src_x1= xmax + 1
    dst_y0= ymin + shift_y
    dst_y1= ymax + 1 + shift_y
    dst_x0= xmin + shift_x
    dst_x1= xmax + 1 + shift_x
    newimage[dst_y0:dst_y1, dst_x0:dst_x1]= image[src_y0:src_y1, src_x0:src_x1]
    newdepth[dst_y0:dst_y1, dst_x0:dst_x1]= depth[src_y0:src_y1, src_x0:src_x1]
    return newimage,newdepth

def image_crop(image, depth, xmin, xmax, ymin, ymax):
    H, W= image.shape[:2]
    max_left_crop= xmin
    max_right_crop= W -xmax-1
    max_up_crop= ymin
    max_down_crop= H -ymax-1
    crop_left= random.randint(0, max_left_crop)
    crop_right= random.randint(0, max_right_crop)
    crop_up= random.randint(0, max_up_crop)
    crop_down= random.randint(0, max_down_crop)
    x0= crop_left
    x1= W-crop_right
    y0= crop_up
    y1= H-crop_down
    newimage= image[y0:y1, x0:x1].copy()
    newdepth= depth[y0:y1, x0:x1].copy()
    return newimage, newdepth

def image_horizontal_flip(img, depth):
    newimg= np.flip(img, axis=1).copy()
    newdepth= np.flip(depth, axis=1).copy()
    return newimg, newdepth

def image_rotation(img, depth, xmin, xmax, ymin, ymax):
    H, W=img.shape[:2]
    angle= random.uniform(-10, 10)
    cx= (xmin+xmax)/2.0
    cy= (ymin+ymax)/2.0
    center=(cx,cy)
    M=cv2.getRotationMatrix2D(center,angle,1.0)
    newimg= cv2.warpAffine(img,M,(W, H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    newdepth= cv2.warpAffine(depth,M,(W, H),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    return newimg, newdepth

def image_shearing(img, depth, xmin, xmax, ymin, ymax):
    H, W=img.shape[:2]
    shear=random.uniform(-0.25, 0.25)
    if random.random()>0.5:
        #Horizontal shear
        M=np.float32([[1, shear, -shear * (ymin + ymax) / 2.0],[0, 1, 0]])
    else:
        #Vertical shear
        M=np.float32([[1, 0, 0],[shear, 1, -shear * (xmin + xmax) / 2.0]])
    newimg= cv2.warpAffine(img,M,(W, H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    newdepth =cv2.warpAffine(depth,M, (W, H),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    return newimg, newdepth

def augment_data(data_list, DATA, options):
    L = len(data_list)
    for index in range(L):
        img_path = os.path.join(DATA, "image", data_list[index] + ".jpg")
        depth_path = os.path.join(DATA, "depth", data_list[index] + ".npy")
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: image not found:{img_path}")
            continue
        img[img < 5]=0
        img=img.astype(np.uint8)
        if not os.path.exists(depth_path):
            print(f"Warning: depth not found:{depth_path}")
            continue
        depth = np.load(depth_path)
        bbox = get_bbox(img)
        if bbox is None:
            print(f"Warning: empty foreground: {img_path}")
            continue
        xmin, xmax, ymin, ymax= bbox
        newimg = copy.deepcopy(img)
        newpathapp = "_aug1"
        augmentations= ["crop", "shift", "flip", "rotation", "shearing"]
        selected_augs=[random.choice(augmentations)]
        if random.random() < options["second_aug_probability"]:
            remaining_augs= [aug for aug in augmentations if aug not in selected_augs]
            selected_augs.append(random.choice(remaining_augs))
        for aug in selected_augs:       
            if aug == "crop":
                newimg,depth=image_crop(newimg, depth, xmin, xmax, ymin, ymax)
                newpathapp+="_c"
            elif aug == "shift":    
                newimg,depth = image_shift(newimg, depth, xmin, xmax, ymin, ymax)
                newpathapp+="_s"
            elif aug == "flip":    
                newimg,depth = image_horizontal_flip(newimg, depth)
                newpathapp+="_f"
            elif aug == "rotation":    
                newimg,depth = image_rotation(newimg, depth, xmin, xmax, ymin, ymax)
                newpathapp+="_r"
            elif aug == "shearing":
                newimg, depth = image_shearing(newimg, depth, xmin, xmax, ymin, ymax)
                newpathapp += "_sh"
            bbox= get_bbox(newimg)
            if bbox is None:
                break
            xmin, xmax, ymin, ymax = bbox

        if newpathapp!="_aug1" and len(newimg)>0:
            '''
            matplotlib.use('TkAgg')
            fig, axs = plt.subplots(3)
            axs[0].imshow(img)
            #ax.add_patch(patches.Rectangle((96,96),422-96,422-96,edgecolor="red",facecolor='none'))
            axs[1].imshow(newimg)
            #ax.add_patch(patches.Rectangle((96,96),422-96,422-96,edgecolor="red",facecolor='none'))
            axs[2].imshow(depth)
            plt.show()
            '''
            print("augmented",img_path,"with",newpathapp)
            img_path = img_path.split(".")[0]+newpathapp+".jpg"
            cv2.imwrite(img_path,newimg)
            depth_path = depth_path.split(".")[0]+newpathapp
            np.save(depth_path,depth)

def image_degradation(item, thresh):
    if item>0 and random.random()<thresh**3:
        return int((item + 3*random.randint(0,255))/4)
    else:
        return item

def image_grayscale(image):
    image= image.astype(np.uint8)
    gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def image_uneven_scaling(img, depth, xmin, xmax, ymin, ymax):
    H, W = img.shape[:2]
    cx= (xmin + xmax) / 2.0
    cy= (ymin + ymax) / 2.0
    alpha = random.uniform(0.7, 1.3)
    if alpha>1:
        beta = random.uniform(0.7, 1)
    else:
        beta = random.uniform(0.7, 1.3)
    growth_ratio = alpha*beta
    newimg = np.zeros_like(img)
    newdepth = np.zeros_like(depth)
    for y in range(ymin, ymax + 1):
        for x in range(xmin, xmax + 1):
            p=(y - cy) * alpha + cy
            q= (x - cx)*beta+cx
            p = int(p)
            q = int(q)
            if 0 <= p<H and 0<=q< W:
                newimg[p, q] = img[y, x]
                newdepth[p, q] = depth[y, x]
    #fix_gaps
    if beta>1:
        for x in range(2,W-3):
            oldv = np.sum(newdepth[ymin:ymax + 1, x])
            newv = np.sum(newdepth[ymin:ymax + 1, x + 1])
            if oldv==0 and newv>0:
                newimg[:,x] = newimg[:,x+1]
                newdepth[:,x] = newdepth[:,x+1]
            elif oldv == 0 and np.sum(newdepth[ymin:ymax + 1, x + 2])>0:
                newimg[:,x] = newimg[:,x+2]
                newdepth[:,x] = newdepth[:,x+2]
    if alpha>1:
        for y in range(2,H-3):
            oldv = np.sum(newdepth[y, xmin:xmax + 1])
            newv = np.sum(newdepth[y + 1, xmin:xmax + 1])
            if oldv==0 and newv>0:
                newimg[y,:] = newimg[y+1,:]
                newdepth[y,:] = newdepth[y+1,:]
            elif oldv == 0 and np.sum(newdepth[y + 2, xmin:xmax + 1])>0:
                newimg[y,:] = newimg[y+2,:]
                newdepth[y,:] = newdepth[y+2,:]
    #adjust depth if size changed
    mask = newdepth > 0
    if np.any(mask):
        newdepth = newdepth.astype(np.float32)
        newdepth[mask] = newdepth[mask] / np.sqrt(growth_ratio)
        # newdepth = newdepth/np.sqrt(growth_ratio) #LESS DEPTH=CLOSER
    return newimg, newdepth

def image_perspective_skewing(img, depth, xmin, xmax, ymin, ymax):
    H, W = img.shape[:2]
    cx = (xmin+xmax) / 2.0
    cy = (ymin+ymax) / 2.0
    alpha = 1
    beta = 0.7
    d = random.uniform(0.85, 1.5)
    axis = xmin#+random.randint(-20,20)
    dfactor = d*(xmax-xmin)
    newimg = np.zeros_like(img)
    
    flipbool = (random.random()>0.5)
    
    hasMask = False
    mask = depth > 0
    if np.any(mask):
        hasMask = True
        newdepth = np.zeros_like(depth)
    else:
        newdepth = copy.deepcopy(depth)
    
    if flipbool:
        img = np.flip(img, axis=1).copy()
        depth = np.flip(depth, axis=1).copy()
        old_xmin= xmin
        old_xmax= xmax
        xmin= W -old_xmax- 1
        xmax= W -old_xmin - 1
        cx= (xmin + xmax)/ 2.0
        axis= xmin
        dfactor= d * (xmax - xmin)
    
    for y in range(ymin, ymax + 1):
        for x in range(xmin, xmax + 1):
            #int(alpha*j)-x0
            p = (y - cy)*(beta + (x - axis)) / dfactor + cy
            q = x
            p = int(p)
            q = int(q)
            if 0 <= p < H and 0 <= q < W:
                newimg[p,q] = img[y, x]
                if hasMask: #adjust mask!
                    factor = (dfactor / max(x - axis + 1, 1))**(1/3)
                    newdepth[p, q] = depth[y, x] * factor
    if flipbool:
        newdepth = np.flip(newdepth, axis=1).copy()
        newimg = np.flip(newimg, axis=1).copy()
    return newimg, newdepth


def augment_data_plus(data_list, DATA, options):
    L = len(data_list)
    for index in range(L):
        img_path = os.path.join(DATA, "image", data_list[index] + ".jpg")
        depth_path = os.path.join(DATA, "depth", data_list[index] + ".npy")
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: image not found:{img_path}")
            continue
        img[img < 5]=0
        img=img.astype(np.uint8)
        # img = np.vectorize(clear_background_noise)(img)
        if not os.path.exists(depth_path):
            print(f"Warning: depth not found:{depth_path}")
            continue
        depth = np.load(depth_path)
        bbox = get_bbox(img)
        if bbox is None:
            print(f"Warning: empty foreground: {img_path}")
            continue
        xmin, xmax, ymin, ymax= bbox   
        newimg = copy.deepcopy(img)
        newpathapp = "_aug2"
        augmentations= ["crop", "shift", "flip", "rotation", "shearing", "dirt","perspective","uneven_scale","grayscale","blur"]
        selected_augs=[random.choice(augmentations)]
        if random.random() < options["second_aug_probability"]:
            remaining_augs= [aug for aug in augmentations if aug not in selected_augs]
            selected_augs.append(random.choice(remaining_augs))
        for aug in selected_augs:       
            if aug == "crop":
                newimg,depth=image_crop(newimg, depth, xmin, xmax, ymin, ymax)
                newpathapp+="_c"
            elif aug == "shift":    
                newimg,depth = image_shift(newimg, depth, xmin, xmax, ymin, ymax)
                newpathapp+="_s"
            elif aug == "flip":    
                newimg,depth = image_horizontal_flip(newimg, depth)
                newpathapp+="_f"
            elif aug == "rotation":    
                newimg,depth = image_rotation(newimg, depth, xmin, xmax, ymin, ymax)
                newpathapp+="_r"
            elif aug == "shearing":
                newimg, depth = image_shearing(newimg, depth, xmin, xmax, ymin, ymax)
                newpathapp += "_sh"
            elif aug == "dirt":   
                newimg = np.vectorize(image_degradation)(newimg,0.3).astype(np.uint8)
                newpathapp+="_d"
            elif aug == "perspective":
                newimg,depth = image_perspective_skewing(newimg, depth, xmin, xmax, ymin, ymax)
                newpathapp+="_p"
            elif aug == "uneven_scale":
                newimg,depth = image_uneven_scaling(newimg, depth, xmin, xmax, ymin, ymax)
                newpathapp+="_us"
            elif aug == "grayscale":
                newimg = image_grayscale(newimg)
                newpathapp+="_g"
            elif aug == "blur":
                newimg = cv2.blur(newimg,(10,10))
                newpathapp+="_b"
            bbox= get_bbox(newimg)
            if bbox is None:
                break
            xmin, xmax, ymin, ymax = bbox

        if newpathapp!="_aug2" and len(newimg)>0:
            print("augmented",img_path,"with",newpathapp)
            img_path = img_path.split(".")[0]+newpathapp+".jpg"
            cv2.imwrite(img_path,newimg)
            depth_path = depth_path.split(".")[0]+newpathapp
            np.save(depth_path,depth)

if __name__ == "__main__":
    ROOT = os.path.abspath(os.path.dirname(__file__))
    DATA = os.path.join(ROOT, 'data')
    image_dir= os.path.join(DATA, "image")
    train_list, validation_list, test_list = read_data(DATA)
    data_list=[os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(".jpg") and "_aug1" not in f and "_aug2" not in f]
    data_list=[path for path in data_list if path in train_list]
    percentage=0.12
    n=max(1, int(len(data_list) * percentage))
    selected_data=random.sample(data_list, n)
    print(f"Selected {n}/{len(data_list)} images for data augmentation")
    options={"second_aug_probability": 0.3}
    augment_data(selected_data, DATA, options)
    augment_data_plus(selected_data, DATA, options)
    