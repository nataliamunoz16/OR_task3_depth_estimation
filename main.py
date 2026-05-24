import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader 
from utils import DataGenerator, read_data, compute_train_max_depth
from unet_task3 import UNet, UNetjoints
from deit_small import DeiTDepth, DeiTDepthjoints
from train import train_model 
from unet_task4 import UNetdropout


ROOT = os.path.abspath(os.path.dirname(__file__))
DATA = os.path.join(ROOT, 'data')
RESULTS = os.path.join(ROOT, 'results') 
PLOTS = os.path.join(RESULTS, 'plots') 
os.makedirs(RESULTS, exist_ok=True) 
os.makedirs(PLOTS, exist_ok=True) 

def build_deit_unfreezing_epochs(n_epochs, n_blocks):
    #builds a schedule for progressively unfreezing DeiT transformer blocks during training
    epochs = [n_epochs // 5, 2 * n_epochs // 5, 3 * n_epochs // 5, 4 * n_epochs // 5]
    start_blocks =[3*n_blocks//4, n_blocks//2, n_blocks//4, 0]
    schedule = {}
    for epoch, start_block in zip(epochs, start_blocks):
        schedule[min(max(epoch, 0),n_epochs-1)] = start_block
    return schedule


def main():
    target_size = (resolution, resolution) if isinstance(resolution, int) else resolution 
    train_list, validation_list, test_list = read_data(DATA)
    max_depth = compute_train_max_depth(train_list, DATA)
    train_dg = DataGenerator(train_list, DATA,  target_size=target_size, max_depth=max_depth, loss_method=loss_method) 
    train_depth_target = "per_image"
    if partition == 'validation': 
        eval_list = validation_list 
        metric_name = 'val' 
    elif partition == 'test': 
        eval_list = test_list 
        metric_name = 'test' 
    else: 
        raise ValueError("partition must be 'validation' or 'test'") 
    eval_dg = DataGenerator(eval_list, DATA, target_size=target_size, max_depth=max_depth, loss_method=loss_method) 
    train_loader = DataLoader(train_dg, batch_size=batch_size, shuffle=shuffle) 
    eval_loader = DataLoader(eval_dg, batch_size=batch_size, shuffle=False) 
    input_shape = (batch_size, 3, target_size[0], target_size[1]) 
    unfreezing_epochs = None
    if loss_method == "multitask" or loss_method=="pose_guided":
        if model_name == 'unet':
            model = UNetjoints(n_class, f1, f2, f3, f4, f5,activation, output_activation)
        elif model_name == 'deit':
            model = DeiTDepthjoints(n_class,target_size,patch_size=deit_patch_size,decoder=deit_decoder,pretrained=pretrained_encoder,output_activation=output_activation)
            if transfer_learning == 'freeze_backbone':
                model.freeze_backbone()
            elif transfer_learning == 'gradual_unfreezing':
                model.freeze_backbone()
                unfreezing_epochs = build_deit_unfreezing_epochs(n_epochs,len(model.encoder.blocks))
            elif transfer_learning != 'none':
                raise ValueError("transfer_learning must be 'none', 'freeze_backbone' or 'gradual_unfreezing'")
        else:
            raise ValueError("model_name must be 'unet' or 'deit'")
    else:
        if model_name == 'unet':
            if not dropout:
                model = UNet(n_class, f1, f2, f3, f4, f5,activation, output_activation)
            else:
                model=UNetdropout(n_class, f1, f2, f3, f4, f5, activation, output_activation, dropout_rate_early=0.3, dropout_rate_late=0.5)
        elif model_name == 'deit':
            model = DeiTDepth(n_class,target_size,patch_size=deit_patch_size,decoder=deit_decoder,pretrained=pretrained_encoder,output_activation=output_activation)
            if transfer_learning == 'freeze_backbone':
                model.freeze_backbone()
            elif transfer_learning == 'gradual_unfreezing':
                model.freeze_backbone()
                unfreezing_epochs = build_deit_unfreezing_epochs(n_epochs,len(model.encoder.blocks))
            elif transfer_learning != 'none':
                raise ValueError("transfer_learning must be 'none', 'freeze_backbone' or 'gradual_unfreezing'")
        else:
            raise ValueError("model_name must be 'unet' or 'deit'") 
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr) 
    return train_model(
        model,
        train_loader,
        eval_loader,
        optimizer,
        device,
        n_epochs,
        input_shape,
        checkpoint,
        metrics_path,
        plot_path,
        metric_name,
        train_depth_target=train_depth_target,
        verbose=verbose,
        loss_method=loss_method,
        gradual_unfreezing=(transfer_learning == 'gradual_unfreezing'),
        unfreezing_epochs=unfreezing_epochs,
        unfreezing_lr=lr,
        use_amp=use_amp
    )


if __name__=='__main__':
    # HYPERPARAMETERS
    lr = 1e-3
    batch_size = 2
    resolution = 384
    n_epochs = 10
    model_name = ['unet', 'deit'][0]
    partition = ['validation', 'test'][0]
    loss_method=None
    use_amp=True
    dropout=False


    # DEIT-SMALL PARAMETERS
    deit_patch_size = [16, 8][0]
    deit_decoder = ['bilinear', 'fpn'][0]
    pretrained_encoder = True 
    transfer_learning = ['none', 'freeze_backbone', 'gradual_unfreezing'][0]

    # UNET2D PARAMETERS
    n_class = 1 
    f1 = 64 
    f2 = 128 
    f3 = 256 
    f4 = 512 
    f5 = 1024 
    activation = 'gelu' 
    output_activation = 'sigmoid' 

    file_name = f"{model_name}_{partition}_lr{lr}_bs{batch_size}_res{resolution}_ep{n_epochs}_lossname{loss_method}_useamp{use_amp}_dropout{dropout}" 
    if model_name == 'deit': 
        file_name = file_name + f"_patch{deit_patch_size}_{deit_decoder}_{transfer_learning}"
    checkpoint = os.path.join(RESULTS, file_name + '_best_model.pt') 
    metrics_path = os.path.join(RESULTS, file_name + '_training_metrics.json') 
    plot_path = os.path.join(PLOTS, file_name + '_loss.png') 
    shuffle = True
    verbose = 1
    main()
