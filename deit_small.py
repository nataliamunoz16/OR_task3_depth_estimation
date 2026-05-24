import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def output_activation_function(name, x):
    name= name.lower()
    if name== 'sigmoid':
        return torch.sigmoid(x)
    if name== 'tanh':
        return torch.tanh(x)
    if name in ['linear', 'none']:
        return x
    raise ValueError("output_activation must be 'sigmoid', 'tanh', 'linear' or 'none'")


class BilinearDecoder(nn.Module):
    def __init__(self, embed_dim, n_class):
        super().__init__()
        self.head= nn.Sequential(nn.Conv2d(embed_dim, 128, kernel_size=3, padding=1),nn.GELU(),nn.Conv2d(128, n_class, kernel_size=1))
    def forward(self, feature, output_size):
        depth= self.head(feature)
        depth= F.interpolate(depth, size=output_size, mode='bilinear', align_corners=False)
        return depth


class LightweightFPNDecoder(nn.Module):
    def __init__(self, embed_dim, n_class, n_features=4, fpn_dim=128):
        super().__init__()
        self.lateral_convs = nn.ModuleList([nn.Conv2d(embed_dim, fpn_dim, kernel_size=1) for _ in range(n_features)])
        self.refine = nn.Sequential(nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1),nn.GELU())
        self.fuse = nn.Sequential(nn.Conv2d(fpn_dim * n_features, fpn_dim, kernel_size=3, padding=1),nn.GELU(), nn.Conv2d(fpn_dim, n_class, kernel_size=1))
    def forward(self, features, output_size):
        pyramid= []
        x = None
        for feature, lateral_conv in reversed(list(zip(features, self.lateral_convs))):
            lateral= lateral_conv(feature)
            if x is not None:
                x= F.interpolate(x, size=lateral.shape[2:], mode='bilinear', align_corners=False)
                lateral= lateral+x
            x =self.refine(lateral)
            pyramid.insert(0, x)
        x = torch.cat(pyramid, dim=1)
        depth =self.fuse(x)
        depth =F.interpolate(depth, size=output_size,mode='bilinear',align_corners=False)
        return depth


class DeiTDepth(nn.Module):
    def __init__(self, n_class, img_size, patch_size=16, decoder='bilinear', pretrained=True, output_activation='sigmoid'):
        super().__init__()
        self.patch_size= patch_size
        self.decoder_type= decoder
        self.output_activation = output_activation
        if patch_size == 16:
            encoder_name= 'deit_small_patch16_224'
        elif patch_size == 8:
            encoder_name= 'vit_small_patch8_224'
        else:
            raise ValueError("patch_size must be 8 or 16")
        self.encoder =timm.create_model(encoder_name, pretrained=pretrained, img_size=img_size, num_classes=0)
        embed_dim =self.encoder.embed_dim
        last_block =len(self.encoder.blocks) - 1
        self.feature_indices =[2,5,8,last_block]
        if decoder== 'bilinear':
            self.decoder= BilinearDecoder(embed_dim, n_class)
        elif decoder== 'fpn':
            self.decoder= LightweightFPNDecoder(embed_dim, n_class, n_features=len(self.feature_indices))
        else:
            raise ValueError("Decoder must be 'bilinear' or 'fpn'")

    def freeze_backbone(self):
        #freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        #unfreeze decoder parameters
        for param in self.decoder.parameters():
            param.requires_grad = True

    def unfreeze_encoder_from(self, start_block):
        #unfreeze the encoder parameters from a certain block onwards
        params = []
        start_block = max(0, min(start_block, len(self.encoder.blocks)))
        modules= [self.encoder] if start_block == 0 else list(self.encoder.blocks[start_block:])
        for module in modules:
            for param in module.parameters():
                if not param.requires_grad:
                    param.requires_grad =True
                    params.append(param)
        return params
    def forward(self, x):
        output_size = x.shape[2:]
        if self.decoder_type == 'bilinear':
            _, features= self.encoder.forward_intermediates(x, indices=[self.feature_indices[-1]], output_fmt='NCHW')
            depth = self.decoder(features[-1], output_size)
        else:
            _, features = self.encoder.forward_intermediates(x, indices=self.feature_indices, output_fmt='NCHW')
            depth = self.decoder(features, output_size)
        depth= output_activation_function(self.output_activation, depth)
        return depth
    

class DeiTDepthjoints(nn.Module):
    def __init__(self, n_class, img_size, patch_size=16, decoder='bilinear', pretrained=True, output_activation='sigmoid'):
        super().__init__()
        self.patch_size= patch_size
        self.decoder_type= decoder
        self.output_activation = output_activation
        if patch_size == 16:
            encoder_name= 'deit_small_patch16_224'
        elif patch_size == 8:
            encoder_name= 'vit_small_patch8_224'
        else:
            raise ValueError("patch_size must be 8 or 16")
        self.encoder = timm.create_model(encoder_name, pretrained=pretrained, img_size=img_size, num_classes=0)
        embed_dim = self.encoder.embed_dim
        last_block = len(self.encoder.blocks) - 1
        self.feature_indices = [2, 5, 8, last_block]
        if decoder == 'bilinear':
            self.depth_decoder  = BilinearDecoder(embed_dim, n_class)
            self.pose_decoder = BilinearDecoder(embed_dim, 14)
        elif decoder == 'fpn':
            self.depth_decoder  = LightweightFPNDecoder(embed_dim, n_class, n_features=len(self.feature_indices))
            self.pose_decoder  = LightweightFPNDecoder(embed_dim, 14, n_features=len(self.feature_indices))
        else:
            raise ValueError("Decoder must be 'bilinear' or 'fpn'")
    
    def freeze_backbone(self):
        #freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        #unfreeze depth decoder parameters
        for param in self.depth_decoder.parameters():
            param.requires_grad = True
        #unfreeze pose decoder parameters
        for param in self.pose_decoder.parameters():
            param.requires_grad = True
    def unfreeze_encoder_from(self, start_block):
        #unfreeze the encoder parameters from a certain block onwards
        params= []
        start_block = max(0, min(start_block, len(self.encoder.blocks)))
        modules= [self.encoder] if start_block == 0 else list(self.encoder.blocks[start_block:])
        for module in modules:
            for param in module.parameters():
                if not param.requires_grad:
                    param.requires_grad = True
                    params.append(param)
        return params

    def forward(self, x):
        output_size = x.shape[2:]
        if self.decoder_type == 'bilinear':
            _, features = self.encoder.forward_intermediates(x, indices=[self.feature_indices[-1]], output_fmt='NCHW')
            feature= features[-1]
            depth= self.depth_decoder(feature, output_size)
            pose_heatmaps = self.pose_decoder(feature, output_size)
        else:
            _, features = self.encoder.forward_intermediates(x, indices=self.feature_indices, output_fmt='NCHW')
            depth = self.depth_decoder(features, output_size)
            pose_heatmaps = self.pose_decoder(features, output_size)
        depth = output_activation_function(self.output_activation, depth)
        return depth, pose_heatmaps