import os
import sys
import cv2
import traceback
import numpy as np
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)
from DataReader.read import DataReader
from DataReader.util import proj, loadInfo, intrinsic, extrinsic


SAMPLES_DIR=os.path.join(ROOT_DIR, 'Samples')
# SAMPLES_DIR = os.path.join(ROOT_DIR, 'cloth3d++_subset', 'cloth3d++_subset')
OUTPUT_DIR=os.path.join(ROOT_DIR, 'data')
MARGIN=10
MIN_PIXELS=5000

reader=DataReader()
def get_crop_box(mask,margin=10,min_pixels=5000):
    # square crop box
    if mask.sum()<min_pixels:
        return None
    ys,xs=np.where(mask)
    xmin,xmax=xs.min(),xs.max()
    ymin,ymax=ys.min(),ys.max()
    cx=(xmin+xmax)/2
    cy=(ymin+ymax)/2
    half=max(cx-xmin,xmax-cx,cy-ymin,ymax-cy)
    half =int(np.ceil(half+margin))
    x1,x2=int(round(cx-half)),int(round(cx+ half))
    y1,y2=int(round(cy-half)),int(round(cy+ half))
    return x1,y1,x2,y2

def quads2tris(F):
    out=[]
    for f in F:
        if len(f)==3: out += [f]
        elif len(f)==4: out += [[f[0],f[1],f[2]],[f[0],f[2],f[3]]]
    return np.array(out, np.int32)

def render_depth(sample,frame,info,shape=(480, 640),max_depth=10.0):
    H,W=shape
    P=reader.read_camera(sample)
    V,F=reader.read_human(sample,frame)
    F=np.array(F)

    for garment in info['outfit']:
        try:
            _V=reader.read_garment_vertices(sample, garment, frame)
            _F= reader.read_garment_topology(sample, garment)
            _F=quads2tris(_F)
            F=np.concatenate((F, _F + V.shape[0]), axis=0)
            V=np.concatenate((V, _V), axis=0)
        except Exception as e:
            continue

    ones=np.ones((V.shape[0], 1), np.float32)
    V_h=np.hstack([V, ones])
    uvw=(P @ V_h.T).T

    depth_map=np.full((H, W), max_depth, dtype=np.float32)

    for tri in F:
        pts_uvw=uvw[tri]
        if np.any(pts_uvw[:, 2] <= 0):
            continue

        u=pts_uvw[:, 0]/pts_uvw[:, 2]
        v=pts_uvw[:, 1]/pts_uvw[:, 2]
        d=pts_uvw[:, 2]

        u_min,u_max=int(np.floor(u.min())),int(np.ceil(u.max()))
        v_min,v_max=int(np.floor(v.min())),int(np.ceil(v.max()))
        u_min=max(u_min, 0)
        u_max=min(u_max,W-1)
        v_min=max(v_min, 0)
        v_max=min(v_max,H-1)

        for pv in range(v_min,v_max+1):
            for pu in range(u_min,u_max+1):
                denom=(v[1]-v[2])*(u[0]-u[2])+(u[2]-u[1])*(v[0]-v[2])
                if abs(denom) < 1e-10: continue
                
                w0=((v[1]-v[2])*(pu-u[2])+(u[2]-u[1])*(pv-v[2]))/denom
                w1=((v[2]-v[0])*(pu-u[2])+(u[0]-u[2])*(pv-v[2]))/denom
                w2=1-w0-w1

                if w0<0 or w1<0 or w2<0: continue

                depth_val=w0*d[0]+w1*d[1]+w2*d[2]

                if depth_val < depth_map[pv, pu]:
                    depth_map[pv,pu]=min(depth_val,max_depth)
    return depth_map


def process_dataset(samples_dir, output_dir):
    img_dir=os.path.join(output_dir,'image')
    depth_dir=os.path.join(output_dir,'depth')
    os.makedirs(img_dir,exist_ok=True)
    os.makedirs(depth_dir,exist_ok=True)

    saved=0
    skipped=0
    for sample in sorted(os.listdir(samples_dir)):
        frames_dir=os.path.join(samples_dir, sample, 'frames')
        if not os.path.isdir(frames_dir):
            continue

        try:
            info = reader.read_info(sample)
        except Exception as e:
            print(f"{sample} not readed")
            traceback.print_exc()
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
            mask=rgba[:, :, 3] > 0
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

            rgb_crop=rgb[y1:y2, x1:x2]
            cv2.imwrite(os.path.join(img_dir,base_name+'.jpg'),rgb_crop,[cv2.IMWRITE_JPEG_QUALITY, 95])

            depth = render_depth(sample, frame_idx, info, shape=(H, W))
            depth_crop    = depth[y1:y2, x1:x2]
            np.save(os.path.join(depth_dir, base_name + '.npy'), depth_crop)

            saved+=1
            if saved%50==0:
                print(f"Saved {saved} frames")

    print(f"Saved {saved} and skipped {skipped}")


if __name__ == "__main__":
    process_dataset(SAMPLES_DIR, OUTPUT_DIR)