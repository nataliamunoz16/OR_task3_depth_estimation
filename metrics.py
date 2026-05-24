import math
import time
import torch
import torch.nn.functional as F


def rmse_metric(y_true, y_pred):
    mask = (y_true > 0).float()
    error = torch.square(y_true - y_pred) * mask
    rmse = torch.sqrt(torch.sum(error) / (torch.sum(mask) + 1e-8))
    return rmse

def compute_normals(depth, mask):
    dx = torch.zeros_like(depth)
    dy = torch.zeros_like(depth)
    dx[:, :, :, :-1]=depth[:, :, :, 1:]-depth[:, :, :, :-1]
    dx[:, :, :, :-1] = torch.where((mask[:, :, :, :-1] > 0) & (mask[:, :, :, 1:] > 0),dx[:, :, :, :-1],torch.zeros_like(dx[:, :, :, :-1]))
    dy[:, :, :-1, :] = depth[:, :, 1:, :] - depth[:, :, :-1, :]
    dy[:, :, :-1, :] = torch.where((mask[:, :, :-1, :] > 0) & (mask[:, :, 1:, :] > 0),dy[:, :, :-1, :],torch.zeros_like(dy[:, :, :-1, :]))
    normals = torch.cat([-dx, -dy, torch.ones_like(depth)], dim=1)
    normals = F.normalize(normals, p=2, dim=1, eps=1e-8)
    return normals

def normal_mae_metric(y_true, y_pred):
    mask = (y_true > 0).float()
    n_true = compute_normals(y_true, mask)
    n_pred = compute_normals(y_pred, mask)
    dot_product = torch.sum(n_true * n_pred, dim=1, keepdim=True)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    angle = torch.acos(dot_product) * (180.0 / math.pi)
    mae = torch.sum(angle * mask) / (torch.sum(mask) + 1e-8)
    return mae

def measure_efficiency(model, input_shape, device):
    model.eval()
    dummy_input = torch.rand(input_shape, device=device)
    params = sum(p.numel() for p in model.parameters())
    memory = 0.0
    warmup = 10
    iters = 50
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for _ in range(warmup):
            model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.time()
        for _ in range(iters):
            model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        total_time = time.time() - start

    if device.type == "cuda":
        memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    fps = (input_shape[0] * iters) / (total_time + 1e-8)
    latency_ms = (total_time / (input_shape[0] * iters)) * 1000
    eff = {"gpu_memory_mb": float(memory), "parameter_count": params, "fps": fps, "latency_ms_per_image": float(latency_ms)}
    return eff

def sobel_gradient_magnitude(x):
    sobel_x= torch.tensor([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]],dtype=torch.float32,device=x.device).view(1, 1, 3, 3)
    sobel_y= torch.tensor([[-1, -2, -1],[0, 0, 0],[1, 2, 1]],dtype=torch.float32,device=x.device).view(1, 1, 3, 3)
    grad_x= F.conv2d(x, sobel_x,padding=1)
    grad_y= F.conv2d(x, sobel_y,padding=1)
    grad= torch.sqrt(grad_x**2+grad_y**2+1e-8)
    return grad

def masked_rmse_metric(y_true, y_pred, extra_mask):
    valid_mask= y_true >0
    final_mask= valid_mask & extra_mask.bool()
    error= torch.square(y_true - y_pred)
    valid_errors= error[final_mask]
    if valid_errors.numel() ==0:
        return y_pred.sum()*0.0
    return torch.sqrt(valid_errors.mean())


def masked_mae_metric(y_true, y_pred, extra_mask):
    valid_mask= y_true > 0
    final_mask= valid_mask & extra_mask.bool()
    error= torch.abs(y_true - y_pred)
    valid_errors= error[final_mask]
    if valid_errors.numel() == 0:
        return y_pred.sum() * 0.0
    return valid_errors.mean()

def edge_and_non_edge_masks(y_true, edge_percentile=0.75):
    valid_mask=y_true>0
    depth_grad=sobel_gradient_magnitude(y_true)
    valid_grad=depth_grad[valid_mask]
    if valid_grad.numel()==0:
        edge_mask= torch.zeros_like(valid_mask)
        non_edge_mask= torch.zeros_like(valid_mask)
        return edge_mask, non_edge_mask
    threshold= torch.quantile(valid_grad, edge_percentile)
    edge_mask= (depth_grad>=threshold) & valid_mask
    non_edge_mask= (depth_grad<threshold) & valid_mask
    return edge_mask, non_edge_mask

def edge_depth_metrics(y_true, y_pred):
    edge_mask, non_edge_mask=edge_and_non_edge_masks(y_true)
    edge_rmse= masked_rmse_metric(y_true, y_pred, edge_mask)
    non_edge_rmse= masked_rmse_metric(y_true, y_pred, non_edge_mask)
    edge_mae= masked_mae_metric(y_true, y_pred, edge_mask)
    non_edge_mae= masked_mae_metric(y_true, y_pred, non_edge_mask)
    return {"edge_rmse": edge_rmse,"non_edge_rmse": non_edge_rmse,"edge_mae": edge_mae,"non_edge_mae": non_edge_mae}

def image_texture_masks(img, valid_mask, textured_percentile=0.75, smooth_percentile=0.25):
    gray= img.mean(dim=1, keepdim=True)
    texture_strength= sobel_gradient_magnitude(gray)
    valid_texture= texture_strength[valid_mask.bool()]
    if valid_texture.numel()==0:
        textured_mask=torch.zeros_like(valid_mask).bool()
        smooth_mask=torch.zeros_like(valid_mask).bool()
        return textured_mask,smooth_mask
    high_threshold= torch.quantile(valid_texture, textured_percentile)
    low_threshold= torch.quantile(valid_texture, smooth_percentile)
    textured_mask= (texture_strength >= high_threshold) & valid_mask.bool()
    smooth_mask= (texture_strength <= low_threshold) & valid_mask.bool()
    return textured_mask, smooth_mask


def textured_smooth_depth_metrics(img, y_true, y_pred):
    valid_mask=y_true>0
    textured_mask, smooth_mask=image_texture_masks(img, valid_mask)
    textured_rmse= masked_rmse_metric(y_true, y_pred, textured_mask)
    smooth_rmse= masked_rmse_metric(y_true, y_pred, smooth_mask)
    textured_mae= masked_mae_metric(y_true, y_pred, textured_mask)
    smooth_mae =masked_mae_metric(y_true, y_pred, smooth_mask)
    return {"textured_rmse": textured_rmse,"smooth_rmse": smooth_rmse,"textured_mae": textured_mae,"smooth_mae": smooth_mae}

def mean_joint_error_metric(pred_xy, gt_xy, joints_valid):
    joints_valid= joints_valid.bool()
    distance= torch.sqrt(torch.sum((pred_xy - gt_xy)**2, dim=2)+1e-8)
    valid_distances= distance[joints_valid]
    if valid_distances.numel() == 0:
        return pred_xy.sum() * 0.0
    return valid_distances.mean()


def pck_metric(pred_xy, gt_xy, joints_valid, threshold=10.0):
    joints_valid= joints_valid.bool()
    distance= torch.sqrt(torch.sum((pred_xy - gt_xy)**2, dim=2)+1e-8)
    valid_distances= distance[joints_valid]
    if valid_distances.numel() == 0:
        return pred_xy.sum()*0.0
    correct= valid_distances<threshold
    return correct.float().mean()


def normalized_mean_joint_error_metric(pred_xy, gt_xy, joints_valid, image_size):
    mje=mean_joint_error_metric(pred_xy, gt_xy, joints_valid)
    return mje/image_size