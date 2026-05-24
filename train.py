import json 
import torch 
import time
import matplotlib 
matplotlib.use("Agg") 
import matplotlib.pyplot as plt 
# from metrics import rmse_metric, normal_mae_metric, measure_efficiency, compute_normals
from metrics import (
    rmse_metric,
    normal_mae_metric,
    measure_efficiency,
    compute_normals,
    edge_depth_metrics,
    textured_smooth_depth_metrics,
    mean_joint_error_metric,
    pck_metric,
    normalized_mean_joint_error_metric
)
from ranking_loss import EdgeguidedRankingLoss
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

pixel_loss = nn.L1Loss(reduction="none")


def masked_mse_loss(y_pred, y_true): 
    mask = (y_true > 0).float() 
    error = torch.square(y_true - y_pred) * mask 
    return torch.sum(error) / (torch.sum(mask) + 1e-8) 


def normal_consistency_loss(y_pred,y_true,valid_mask):
    pred_normals=compute_normals(y_pred, valid_mask.float())
    true_normals=compute_normals(y_true, valid_mask.float())
    loss=pixel_loss(pred_normals, true_normals)
    valid_mask=valid_mask.bool()
    loss_valid_pixels=loss.permute(0, 2, 3, 1)[valid_mask.squeeze(1)]
    return loss_valid_pixels.mean()

def scale_invariant_depth_loss(y_pred, y_true, valid_mask, lambda_=0.5):
    valid_mask=valid_mask.bool()
    y_pred_valid=y_pred[valid_mask]
    y_true_valid=y_true[valid_mask]
    d=torch.log(y_pred_valid+1e-8)-torch.log(y_true_valid+1e-8)
    loss=torch.mean(d**2)-lambda_*(torch.mean(d)**2)
    return loss

def masked_l1_loss(y_pred,y_true,valid_mask):
    valid_mask=valid_mask.bool()
    l1_error=torch.abs(y_pred - y_true)
    valid_errors=l1_error[valid_mask]
    return valid_errors.mean()

def soft_argmax_2d(heatmaps):
    batch_size, num_joints, height, width=heatmaps.shape
    heatmaps_flat= heatmaps.reshape(batch_size, num_joints, height * width)
    probabilities= F.softmax(heatmaps_flat, dim=2)
    x_coordinates=torch.arange(width,device=heatmaps.device).float()
    y_coordinates=torch.arange(height,device=heatmaps.device).float()
    grid_y, grid_x= torch.meshgrid(y_coordinates,x_coordinates,indexing="ij")
    grid_x= grid_x.reshape(-1)
    grid_y= grid_y.reshape(-1)
    predicted_x=torch.sum(probabilities*grid_x,dim=2)
    predicted_y=torch.sum(probabilities*grid_y,dim=2)
    predicted_xy = torch.stack([predicted_x, predicted_y], dim=2)
    return predicted_xy

def pose_l1_loss(pred_xy, gt_xy, joints_valid):
    joints_valid= joints_valid.bool()
    l1_error=torch.abs(pred_xy-gt_xy)
    valid_mask=joints_valid.unsqueeze(-1)
    valid_l1_error=l1_error[valid_mask.expand_as(l1_error)]
    if valid_l1_error.numel()==0:
        return pred_xy.sum()*0.0
    return valid_l1_error.mean()

def sample_depth_at_joints(depth_map, joints_xy):
    """
    Samples depth values at joint coordinates using bilinear interpolation
    """
    B,_,H,W=depth_map.shape
    x =joints_xy[:, :, 0]
    y =joints_xy[:, :, 1]
    #convert pixel coordinates to normalized coordinates in [-1, 1]
    x_norm =2.0*x/max(W-1, 1)-1.0
    y_norm =2.0*y/max(H-1,1)-1.0
    grid= torch.stack([x_norm, y_norm], dim=-1)
    grid= grid.unsqueeze(2)
    sampled= F.grid_sample(depth_map,grid,mode="bilinear",padding_mode="zeros",align_corners=True)
    sampled= sampled.squeeze(1).squeeze(-1)
    return sampled

def skeleton_depth_loss(pred_depth, gt_depth, joints_xy, joints_valid):
    """
    Penalizes depth errors specifically at valid body joints
    """
    joints_valid= joints_valid.bool()
    pred_joint_depth= sample_depth_at_joints(pred_depth, joints_xy)
    gt_joint_depth= sample_depth_at_joints(gt_depth, joints_xy)
    valid_depth= gt_joint_depth> 0
    final_mask= joints_valid & valid_depth
    if final_mask.sum()==0:
        return pred_depth.sum()*0.0
    error = torch.abs(pred_joint_depth-gt_joint_depth)
    return error[final_mask].mean()

def train_model(model, train_loader, eval_loader, optimizer, device, n_epochs, input_shape, checkpoint, metrics_path, plot_path, metric_name, train_depth_target="per_image", verbose=1, loss_method=None, gradual_unfreezing=False, unfreezing_epochs=None, unfreezing_lr=None, use_amp=False): 
    history = {"epochs": []} 
    best_eval_rmse = float("inf") 

    scaler = torch.amp.GradScaler("cuda" ,enabled=use_amp)

    start=time.time()
    if loss_method == "ranking":
        rank_criterion= EdgeguidedRankingLoss(point_pairs=10000, sigma=0.03, alpha=1.0, mask_value=0.0).to(device)
    for epoch in range(n_epochs): 
        print(f"Epoch {epoch + 1}/{n_epochs}")
        if gradual_unfreezing and unfreezing_epochs and epoch in unfreezing_epochs:
            start_block = unfreezing_epochs[epoch]
            new_params = model.unfreeze_encoder_from(start_block)
            if new_params:
                optimizer.add_param_group({"params": new_params, "lr": unfreezing_lr or optimizer.param_groups[0]["lr"]})
                print(f"Unfroze encoder from block {start_block}")
        model.train() 
        train_loss = 0.0 
        train_rmse_global = 0.0
        train_normal_global = 0.0
        train_rmse_per_image = 0.0
        train_normal_per_image = 0.0 
        train_edge_rmse = 0.0
        train_non_edge_rmse = 0.0
        train_edge_mae = 0.0
        train_non_edge_mae = 0.0
        train_textured_rmse = 0.0
        train_smooth_rmse = 0.0
        train_textured_mae = 0.0
        train_smooth_mae = 0.0
        train_joint_mje = 0.0
        train_joint_nmje = 0.0
        train_pck_10 = 0.0
        train_pck_20 = 0.0
        
        
        typ = torch.float32
        if use_amp:
            typ = torch.float16

        for img, dpt_dict in train_loader: 
            with torch.autocast(device.type if isinstance(device, torch.device) else device, dtype=typ):
                img = img.to(device) 
                dpt_global = dpt_dict["global"].to(device)
                dpt_per_image = dpt_dict["per_image"].to(device)
                if train_depth_target=="global":
                    dpt= dpt_global
                elif train_depth_target=="per_image":
                    dpt= dpt_per_image
                optimizer.zero_grad() 
                if loss_method=="multitask" or loss_method=="pose_guided":
                    pred_depth, pose_heatmaps= model(img)
                    pred= pred_depth
                else:
                    pred= model(img) 
                if loss_method is None:
                    loss = masked_mse_loss(pred, dpt) 
                elif loss_method == "ranking":
                    ranking_target= dpt_per_image
                    valid_mask= dpt_global > 0
                    loss= rank_criterion(pred, ranking_target, img, valid_mask)
                elif loss_method == "normal":
                    normal_target= dpt_per_image
                    valid_mask =dpt_global>0
                    loss =normal_consistency_loss(pred, normal_target, valid_mask)
                elif loss_method == "scale_invariant":
                    si_target=dpt_global
                    valid_mask=dpt_global>0
                    loss=scale_invariant_depth_loss(pred,si_target,valid_mask,lambda_=0.5)
                elif loss_method == "multitask" or loss_method=="pose_guided":
                    gt_xy= dpt_dict["joints_xy"].to(device)
                    joints_valid= dpt_dict["joints_valid"].to(device)
                    valid_mask=dpt_global>0
                    pred_xy= soft_argmax_2d(pose_heatmaps)
                    depth_loss= masked_l1_loss(pred_depth, dpt, valid_mask)
                    joint_loss= pose_l1_loss(pred_xy, gt_xy, joints_valid)
                    if loss_method=="multitask":
                        loss=depth_loss+0.01*joint_loss
                    elif loss_method=="pose_guided":
                        skel_loss=skeleton_depth_loss(pred_depth, dpt, gt_xy, joints_valid)
                        loss=depth_loss+0.01*joint_loss+0.1*skel_loss
            if use_amp:
                scaler.scale(loss).backward()           
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                loss.backward()
                optimizer.step()

            with torch.no_grad(): 
                train_loss += loss.item() 
                train_rmse_global += rmse_metric(dpt_global, pred).item()
                train_normal_global += normal_mae_metric(dpt_global, pred).item()
                train_rmse_per_image += rmse_metric(dpt_per_image, pred).item()
                train_normal_per_image += normal_mae_metric(dpt_per_image, pred).item() 
                
                edge_metrics = edge_depth_metrics(dpt_per_image, pred)
                train_edge_rmse +=edge_metrics["edge_rmse"].item()
                train_non_edge_rmse+= edge_metrics["non_edge_rmse"].item()
                train_edge_mae+= edge_metrics["edge_mae"].item()
                train_non_edge_mae+= edge_metrics["non_edge_mae"].item()

                texture_metrics= textured_smooth_depth_metrics(img, dpt_per_image, pred)
                train_textured_rmse+= texture_metrics["textured_rmse"].item()
                train_smooth_rmse+= texture_metrics["smooth_rmse"].item()
                train_textured_mae+= texture_metrics["textured_mae"].item()
                train_smooth_mae+= texture_metrics["smooth_mae"].item()

                if loss_method == "multitask" or loss_method=="pose_guided":
                    train_joint_mje+= mean_joint_error_metric(pred_xy, gt_xy, joints_valid).item()
                    train_joint_nmje+= normalized_mean_joint_error_metric(pred_xy, gt_xy, joints_valid, image_size=pred.shape[-1]).item()
                    train_pck_10+= pck_metric(pred_xy, gt_xy, joints_valid, threshold=10.0).item()
                    train_pck_20+= pck_metric(pred_xy, gt_xy, joints_valid, threshold=20.0).item()
        train_loss= train_loss / len(train_loader)
        train_rmse_global/= len(train_loader)
        train_normal_global/= len(train_loader)
        train_rmse_per_image/= len(train_loader)
        train_normal_per_image/= len(train_loader)
        train_edge_rmse /= len(train_loader)
        train_non_edge_rmse /= len(train_loader)
        train_edge_mae /= len(train_loader)
        train_non_edge_mae /= len(train_loader)
        train_textured_rmse /= len(train_loader)
        train_smooth_rmse /= len(train_loader)
        train_textured_mae /= len(train_loader)
        train_smooth_mae /= len(train_loader)
        if loss_method == "multitask" or loss_method=="pose_guided":
            train_joint_mje /= len(train_loader)
            train_joint_nmje /= len(train_loader)
            train_pck_10 /= len(train_loader)
            train_pck_20 /= len(train_loader)

        model.eval() 
        eval_loss = 0.0
        eval_rmse_global = 0.0
        eval_normal_global = 0.0
        eval_rmse_per_image = 0.0
        eval_normal_per_image = 0.0

        eval_edge_rmse = 0.0
        eval_non_edge_rmse = 0.0
        eval_edge_mae = 0.0
        eval_non_edge_mae = 0.0

        eval_textured_rmse = 0.0
        eval_smooth_rmse = 0.0
        eval_textured_mae = 0.0
        eval_smooth_mae = 0.0

        eval_joint_mje = 0.0
        eval_joint_nmje = 0.0
        eval_pck_10 = 0.0
        eval_pck_20 = 0.0
        with torch.no_grad(): 
            for img, dpt_dict in eval_loader: 
                img = img.to(device) 
                dpt_global = dpt_dict["global"].to(device)
                dpt_per_image = dpt_dict["per_image"].to(device)
                if train_depth_target == "global":
                    dpt= dpt_global
                elif train_depth_target == "per_image":
                    dpt =dpt_per_image
                if loss_method=="multitask" or loss_method=="pose_guided":
                    pred_depth, pose_heatmaps= model(img)
                    pred= pred_depth
                else:
                    pred= model(img)
                # loss = masked_mse_loss(pred, dpt) 
                if loss_method is None:
                    loss = masked_mse_loss(pred, dpt) 
                elif loss_method == "ranking":
                    ranking_target= dpt_per_image
                    valid_mask= dpt_global > 0
                    loss= rank_criterion(pred, ranking_target, img, valid_mask)
                elif loss_method == "normal":
                    normal_target= dpt_per_image
                    valid_mask =dpt_global>0
                    loss =normal_consistency_loss(pred, normal_target, valid_mask)
                elif loss_method == "scale_invariant":
                    si_target=dpt_global
                    valid_mask=dpt_global>0
                    loss=scale_invariant_depth_loss(pred,si_target,valid_mask,lambda_=0.5)
                elif loss_method == "multitask" or loss_method=="pose_guided":
                    gt_xy= dpt_dict["joints_xy"].to(device)
                    joints_valid= dpt_dict["joints_valid"].to(device)
                    valid_mask=dpt_global>0
                    pred_xy= soft_argmax_2d(pose_heatmaps)
                    depth_loss= masked_l1_loss(pred_depth, dpt, valid_mask)
                    joint_loss= pose_l1_loss(pred_xy, gt_xy, joints_valid)
                    if loss_method=="multitask":
                        loss=depth_loss+0.01*joint_loss
                    elif loss_method=="pose_guided":
                        skel_loss=skeleton_depth_loss(pred_depth, dpt, gt_xy, joints_valid)
                        loss=depth_loss+0.01*joint_loss+0.1*skel_loss
                eval_loss += loss.item() 
                eval_rmse_global += rmse_metric(dpt_global, pred).item()
                eval_normal_global += normal_mae_metric(dpt_global, pred).item()
                eval_rmse_per_image += rmse_metric(dpt_per_image, pred).item()
                eval_normal_per_image += normal_mae_metric(dpt_per_image, pred).item() 
                edge_metrics = edge_depth_metrics(dpt_per_image, pred)
                eval_edge_rmse +=edge_metrics["edge_rmse"].item()
                eval_non_edge_rmse +=edge_metrics["non_edge_rmse"].item()
                eval_edge_mae += edge_metrics["edge_mae"].item()
                eval_non_edge_mae += edge_metrics["non_edge_mae"].item()

                texture_metrics =textured_smooth_depth_metrics(img, dpt_per_image, pred)
                eval_textured_rmse+= texture_metrics["textured_rmse"].item()
                eval_smooth_rmse+= texture_metrics["smooth_rmse"].item()
                eval_textured_mae+= texture_metrics["textured_mae"].item()
                eval_smooth_mae += texture_metrics["smooth_mae"].item()

                if loss_method == "multitask" or loss_method=="pose_guided":
                    eval_joint_mje += mean_joint_error_metric(pred_xy, gt_xy, joints_valid).item()
                    eval_joint_nmje+=normalized_mean_joint_error_metric(pred_xy, gt_xy, joints_valid, image_size=pred.shape[-1]).item()
                    eval_pck_10+= pck_metric(pred_xy, gt_xy, joints_valid, threshold=10.0).item()
                    eval_pck_20+= pck_metric(pred_xy, gt_xy, joints_valid, threshold=20.0).item()
        eval_loss = eval_loss / len(eval_loader) 
        eval_rmse_global /= len(eval_loader)
        eval_normal_global /= len(eval_loader)
        eval_rmse_per_image /= len(eval_loader)
        eval_normal_per_image /= len(eval_loader)
        eval_edge_rmse /= len(eval_loader)
        eval_non_edge_rmse /= len(eval_loader)
        eval_edge_mae /= len(eval_loader)
        eval_non_edge_mae /= len(eval_loader)

        eval_textured_rmse /= len(eval_loader)
        eval_smooth_rmse /= len(eval_loader)
        eval_textured_mae /= len(eval_loader)
        eval_smooth_mae /= len(eval_loader)

        if loss_method == "multitask" or loss_method=="pose_guided":
            eval_joint_mje/= len(eval_loader)
            eval_joint_nmje/= len(eval_loader)
            eval_pck_10/= len(eval_loader)
            eval_pck_20/= len(eval_loader) 

        # epoch_data = {"epoch": epoch + 1, "train_loss": train_loss, f"{metric_name}_loss": eval_loss, 
        #               "train_rmse_global": train_rmse_global, f"{metric_name}_rmse_global": eval_rmse_global, "train_normal_mae_global": train_normal_global, 
        #               f"{metric_name}_normal_mae_global": eval_normal_global, "train_rmse_per_image": train_rmse_per_image, f"{metric_name}_rmse_per_image": eval_rmse_per_image, "train_normal_mae_per_image": train_normal_per_image, 
        #               f"{metric_name}_normal_mae_per_image": eval_normal_per_image, "efficiency": measure_efficiency(model, input_shape, device)} 
        epoch_data = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            f"{metric_name}_loss": eval_loss,
            "train_rmse_global": train_rmse_global,
            f"{metric_name}_rmse_global": eval_rmse_global,
            "train_normal_mae_global": train_normal_global,
            f"{metric_name}_normal_mae_global": eval_normal_global,
            "train_rmse_per_image": train_rmse_per_image,
            f"{metric_name}_rmse_per_image": eval_rmse_per_image,
            "train_normal_mae_per_image": train_normal_per_image,
            f"{metric_name}_normal_mae_per_image": eval_normal_per_image,
            "train_edge_rmse": train_edge_rmse,
            f"{metric_name}_edge_rmse": eval_edge_rmse,
            "train_non_edge_rmse": train_non_edge_rmse,
            f"{metric_name}_non_edge_rmse": eval_non_edge_rmse,
            "train_edge_mae": train_edge_mae,
            f"{metric_name}_edge_mae": eval_edge_mae,
            "train_non_edge_mae": train_non_edge_mae,
            f"{metric_name}_non_edge_mae": eval_non_edge_mae,
            "train_textured_rmse": train_textured_rmse,
            f"{metric_name}_textured_rmse": eval_textured_rmse,
            "train_smooth_rmse": train_smooth_rmse,
            f"{metric_name}_smooth_rmse": eval_smooth_rmse,
            "train_textured_mae": train_textured_mae,
            f"{metric_name}_textured_mae": eval_textured_mae,
            "train_smooth_mae": train_smooth_mae,
            f"{metric_name}_smooth_mae": eval_smooth_mae,
            "efficiency": measure_efficiency(model, input_shape, device)
        }
        if loss_method == "multitask" or loss_method=="pose_guided":
            epoch_data.update({
                "train_joint_mje": train_joint_mje,
                f"{metric_name}_joint_mje": eval_joint_mje,
                "train_joint_nmje": train_joint_nmje,
                f"{metric_name}_joint_nmje": eval_joint_nmje,
                "train_pck_10": train_pck_10,
                f"{metric_name}_pck_10": eval_pck_10,
                "train_pck_20": train_pck_20,
                f"{metric_name}_pck_20": eval_pck_20,
            })
        history["epochs"].append(epoch_data) 

        with open(metrics_path, "w") as f: 
            json.dump(history, f, indent=4) 

        if eval_rmse_per_image < best_eval_rmse: 
            best_eval_rmse = eval_rmse_per_image 
            torch.save(model.state_dict(), checkpoint) 

        if verbose:
            print(
                f"Epoch {epoch + 1}/{n_epochs} "
                f"- train_loss: {train_loss:.4f} "
                f"- {metric_name}_loss: {eval_loss:.4f} "
                f"- {metric_name}_rmse_global: {eval_rmse_global:.4f} "
                f"- {metric_name}_normal_mae_global: {eval_normal_global:.4f} "
                f"- {metric_name}_rmse_per_image: {eval_rmse_per_image:.4f} "
                f"- {metric_name}_normal_mae_per_image: {eval_normal_per_image:.4f}")
    epochs = [epoch_data["epoch"] for epoch_data in history["epochs"]] 
    train_losses = [epoch_data["train_loss"] for epoch_data in history["epochs"]] 
    eval_losses = [epoch_data[f"{metric_name}_loss"] for epoch_data in history["epochs"]] 
    plt.figure(figsize=(7, 5)) 
    plt.plot(epochs, train_losses, marker="o", label="train_loss", color='steelblue') 
    plt.plot(epochs, eval_losses, marker="o", label=f"{metric_name}_loss", color='lightblue') 
    plt.xlabel("Epoch") 
    plt.ylabel("Loss") 
    plt.title("TRAINING LOSS", fontweight="bold") 
    plt.legend() 
    plt.tight_layout() 
    plt.savefig(plot_path) 
    plt.close() 

    print("Training completed!")
    print(time.time()-start)
    return 0 