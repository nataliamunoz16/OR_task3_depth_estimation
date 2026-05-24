import torch
import torch.nn as nn
import torch.nn.functional as F


def activation_function(name, input):
    dict_map={"relu": torch.nn.functional.relu, "gelu": torch.nn.functional.gelu, "elu":torch.nn.functional.elu, "softmax":torch.nn.functional.softmax, "tanh": torch.nn.functional.tanh, "tahn": torch.nn.functional.tanh, "sigmoid": torch.nn.functional.sigmoid, "silu":torch.nn.functional.silu}
    return dict_map[name.lower()](input)


class UNet(nn.Module):
    def __init__(self, n_class, f1, f2, f3, f4, f5, activation, output_activation):
        super().__init__()
        self.activation=activation
        self.output_activation=output_activation

        # Encoder
        self.e11 = nn.Conv2d(3, f1, kernel_size=3, padding=1)
        self.e11_bn = nn.BatchNorm2d(f1)
        self.e12 = nn.Conv2d(f1, f1, kernel_size=3, padding=1)
        self.e12_bn = nn.BatchNorm2d(f1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e21 = nn.Conv2d(f1, f2, kernel_size=3, padding=1)
        self.e21_bn = nn.BatchNorm2d(f2)
        self.e22 = nn.Conv2d(f2, f2, kernel_size=3, padding=1)
        self.e22_bn = nn.BatchNorm2d(f2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e31 = nn.Conv2d(f2, f3, kernel_size=3, padding=1)
        self.e31_bn = nn.BatchNorm2d(f3)
        self.e32 = nn.Conv2d(f3, f3, kernel_size=3, padding=1)
        self.e32_bn = nn.BatchNorm2d(f3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e41 = nn.Conv2d(f3, f4, kernel_size=3, padding=1)
        self.e41_bn = nn.BatchNorm2d(f4)
        self.e42 = nn.Conv2d(f4, f4, kernel_size=3, padding=1)
        self.e42_bn = nn.BatchNorm2d(f4)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e51 = nn.Conv2d(f4, f5, kernel_size=3, padding=1)
        self.e51_bn = nn.BatchNorm2d(f5)
        self.e52 = nn.Conv2d(f5, f5, kernel_size=3, padding=1)
        self.e52_bn = nn.BatchNorm2d(f5)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(f5, f4, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(f5, f4, kernel_size=3, padding=1)
        self.d11_bn = nn.BatchNorm2d(f4)
        self.d12 = nn.Conv2d(f4, f4, kernel_size=3, padding=1)
        self.d12_bn = nn.BatchNorm2d(f4)

        self.upconv2 = nn.ConvTranspose2d(f4, f3, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(f4, f3, kernel_size=3, padding=1)
        self.d21_bn = nn.BatchNorm2d(f3)
        self.d22 = nn.Conv2d(f3, f3, kernel_size=3, padding=1)
        self.d22_bn = nn.BatchNorm2d(f3)

        self.upconv3 = nn.ConvTranspose2d(f3, f2, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(f3, f2, kernel_size=3, padding=1)
        self.d31_bn = nn.BatchNorm2d(f2)
        self.d32 = nn.Conv2d(f2, f2, kernel_size=3, padding=1)
        self.d32_bn = nn.BatchNorm2d(f2)

        self.upconv4 = nn.ConvTranspose2d(f2, f1, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(f2, f1, kernel_size=3, padding=1)
        self.d41_bn = nn.BatchNorm2d(f1)
        self.d42 = nn.Conv2d(f1, f1, kernel_size=3, padding=1)
        self.d42_bn = nn.BatchNorm2d(f1)

        self.outconv = nn.Conv2d(f1, n_class, kernel_size=1)

    def forward(self, x):
        # Encoder
        xe11=activation_function(self.activation, self.e11_bn(self.e11(x)))
        xe12=activation_function(self.activation, self.e12_bn(self.e12(xe11)))
        xp1 = self.pool1(xe12)

        xe21=activation_function(self.activation, self.e21_bn(self.e21(xp1)))
        xe22=activation_function(self.activation, self.e22_bn(self.e22(xe21)))
        xp2 = self.pool2(xe22)

        xe31=activation_function(self.activation, self.e31_bn(self.e31(xp2)))
        xe32=activation_function(self.activation, self.e32_bn(self.e32(xe31)))
        xp3 = self.pool3(xe32)

        xe41=activation_function(self.activation, self.e41_bn(self.e41(xp3)))
        xe42=activation_function(self.activation, self.e42_bn(self.e42(xe41)))
        xp4 = self.pool4(xe42)

        xe51=activation_function(self.activation, self.e51_bn(self.e51(xp4)))
        xe52=activation_function(self.activation, self.e52_bn(self.e52(xe51)))

        # Decoder
        xu1 = self.upconv1(xe52)
        if xu1.shape[2:] != xe42.shape[2:]:
            xu1 = F.interpolate(xu1, size=xe42.shape[2:], mode='bilinear', align_corners=False)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11=activation_function(self.activation, self.d11_bn(self.d11(xu11)))
        xd12=activation_function(self.activation, self.d12_bn(self.d12(xd11)))

        xu2 = self.upconv2(xd12)
        if xu2.shape[2:] != xe32.shape[2:]:
            xu2 = F.interpolate(xu2, size=xe32.shape[2:], mode='bilinear', align_corners=False)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21=activation_function(self.activation, self.d21_bn(self.d21(xu22)))
        xd22=activation_function(self.activation, self.d22_bn(self.d22(xd21)))

        xu3 = self.upconv3(xd22)
        if xu3.shape[2:] != xe22.shape[2:]:
            xu3 = F.interpolate(xu3, size=xe22.shape[2:], mode='bilinear', align_corners=False)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31=activation_function(self.activation, self.d31_bn(self.d31(xu33)))
        xd32=activation_function(self.activation, self.d32_bn(self.d32(xd31)))

        xu4 = self.upconv4(xd32)
        if xu4.shape[2:] != xe12.shape[2:]:
            xu4 = F.interpolate(xu4, size=xe12.shape[2:], mode='bilinear', align_corners=False)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41=activation_function(self.activation, self.d41_bn(self.d41(xu44)))
        xd42=activation_function(self.activation, self.d42_bn(self.d42(xd41)))

        out = self.outconv(xd42)
        out=activation_function(self.output_activation, out)
        return out


class UNetjoints(nn.Module):
    def __init__(self, n_class, f1, f2, f3, f4, f5, activation, output_activation):
        super().__init__()
        self.activation=activation
        self.output_activation=output_activation

        # Encoder
        self.e11 = nn.Conv2d(3, f1, kernel_size=3, padding=1)
        self.e11_bn = nn.BatchNorm2d(f1)
        self.e12 = nn.Conv2d(f1, f1, kernel_size=3, padding=1)
        self.e12_bn = nn.BatchNorm2d(f1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e21 = nn.Conv2d(f1, f2, kernel_size=3, padding=1)
        self.e21_bn = nn.BatchNorm2d(f2)
        self.e22 = nn.Conv2d(f2, f2, kernel_size=3, padding=1)
        self.e22_bn = nn.BatchNorm2d(f2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e31 = nn.Conv2d(f2, f3, kernel_size=3, padding=1)
        self.e31_bn = nn.BatchNorm2d(f3)
        self.e32 = nn.Conv2d(f3, f3, kernel_size=3, padding=1)
        self.e32_bn = nn.BatchNorm2d(f3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e41 = nn.Conv2d(f3, f4, kernel_size=3, padding=1)
        self.e41_bn = nn.BatchNorm2d(f4)
        self.e42 = nn.Conv2d(f4, f4, kernel_size=3, padding=1)
        self.e42_bn = nn.BatchNorm2d(f4)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e51 = nn.Conv2d(f4, f5, kernel_size=3, padding=1)
        self.e51_bn = nn.BatchNorm2d(f5)
        self.e52 = nn.Conv2d(f5, f5, kernel_size=3, padding=1)
        self.e52_bn = nn.BatchNorm2d(f5)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(f5, f4, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(f5, f4, kernel_size=3, padding=1)
        self.d11_bn = nn.BatchNorm2d(f4)
        self.d12 = nn.Conv2d(f4, f4, kernel_size=3, padding=1)
        self.d12_bn = nn.BatchNorm2d(f4)

        self.upconv2 = nn.ConvTranspose2d(f4, f3, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(f4, f3, kernel_size=3, padding=1)
        self.d21_bn = nn.BatchNorm2d(f3)
        self.d22 = nn.Conv2d(f3, f3, kernel_size=3, padding=1)
        self.d22_bn = nn.BatchNorm2d(f3)

        self.upconv3 = nn.ConvTranspose2d(f3, f2, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(f3, f2, kernel_size=3, padding=1)
        self.d31_bn = nn.BatchNorm2d(f2)
        self.d32 = nn.Conv2d(f2, f2, kernel_size=3, padding=1)
        self.d32_bn = nn.BatchNorm2d(f2)

        self.upconv4 = nn.ConvTranspose2d(f2, f1, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(f2, f1, kernel_size=3, padding=1)
        self.d41_bn = nn.BatchNorm2d(f1)
        self.d42 = nn.Conv2d(f1, f1, kernel_size=3, padding=1)
        self.d42_bn = nn.BatchNorm2d(f1)

        self.depth_head = nn.Conv2d(f1, 1, kernel_size=1)
        self.pose_head = nn.Conv2d(f1, 14, kernel_size=1)

    def forward(self, x):
        # Encoder
        xe11=activation_function(self.activation, self.e11_bn(self.e11(x)))
        xe12=activation_function(self.activation, self.e12_bn(self.e12(xe11)))
        xp1 = self.pool1(xe12)

        xe21=activation_function(self.activation, self.e21_bn(self.e21(xp1)))
        xe22=activation_function(self.activation, self.e22_bn(self.e22(xe21)))
        xp2 = self.pool2(xe22)

        xe31=activation_function(self.activation, self.e31_bn(self.e31(xp2)))
        xe32=activation_function(self.activation, self.e32_bn(self.e32(xe31)))
        xp3 = self.pool3(xe32)

        xe41=activation_function(self.activation, self.e41_bn(self.e41(xp3)))
        xe42=activation_function(self.activation, self.e42_bn(self.e42(xe41)))
        xp4 = self.pool4(xe42)

        xe51=activation_function(self.activation, self.e51_bn(self.e51(xp4)))
        xe52=activation_function(self.activation, self.e52_bn(self.e52(xe51)))

        # Decoder
        xu1 = self.upconv1(xe52)
        if xu1.shape[2:] != xe42.shape[2:]:
            xu1 = F.interpolate(xu1, size=xe42.shape[2:], mode='bilinear', align_corners=False)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11=activation_function(self.activation, self.d11_bn(self.d11(xu11)))
        xd12=activation_function(self.activation, self.d12_bn(self.d12(xd11)))

        xu2 = self.upconv2(xd12)
        if xu2.shape[2:] != xe32.shape[2:]:
            xu2 = F.interpolate(xu2, size=xe32.shape[2:], mode='bilinear', align_corners=False)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21=activation_function(self.activation, self.d21_bn(self.d21(xu22)))
        xd22=activation_function(self.activation, self.d22_bn(self.d22(xd21)))

        xu3 = self.upconv3(xd22)
        if xu3.shape[2:] != xe22.shape[2:]:
            xu3 = F.interpolate(xu3, size=xe22.shape[2:], mode='bilinear', align_corners=False)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31=activation_function(self.activation, self.d31_bn(self.d31(xu33)))
        xd32=activation_function(self.activation, self.d32_bn(self.d32(xd31)))

        xu4 = self.upconv4(xd32)
        if xu4.shape[2:] != xe12.shape[2:]:
            xu4 = F.interpolate(xu4, size=xe12.shape[2:], mode='bilinear', align_corners=False)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41=activation_function(self.activation, self.d41_bn(self.d41(xu44)))
        xd42=activation_function(self.activation, self.d42_bn(self.d42(xd41)))

        depth= self.depth_head(xd42)
        depth= activation_function(self.output_activation, depth)
        pose_heatmaps= self.pose_head(xd42)
        return depth, pose_heatmaps