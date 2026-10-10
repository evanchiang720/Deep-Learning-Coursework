import torch
import torch.nn as nn
import torch.nn.functional as F

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel Attention
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention
        avg_out = self.fc(F.adaptive_avg_pool2d(x, 1))
        max_out = self.fc(F.adaptive_max_pool2d(x, 1))
        x = x * self.sigmoid(avg_out + max_out)
        # Spatial Attention
        avg_s = torch.mean(x, dim=1, keepdim=True)
        max_s, _ = torch.max(x, dim=1, keepdim=True)
        x = x * self.sigmoid(self.conv_spatial(torch.cat([avg_s, max_s], dim=1)))
        return x

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.ReLU = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        x_out = self.conv1(x)
        x_out = self.norm1(x_out)
        x_out = self.ReLU(x_out)
        x_out = self.conv2(x_out)
        x_out = self.norm2(x_out)
        if self.downsample is not None:
            identity = self.downsample(x)
        x_out += identity
        x_out = self.ReLU(x_out)
        return x_out

def make_layer(in_channels, out_channels, num, stride=1):
    downsample = None
    if stride != 1 or in_channels != out_channels:
        downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    layers = []
    layers.append(Block(in_channels=in_channels, out_channels=out_channels, stride=stride, downsample=downsample))
    for i in range(1, num):
        layers.append(Block(in_channels=out_channels, out_channels=out_channels))
    return nn.Sequential(*layers)

class Resnet34_Unet(nn.Module):
    def __init__(self, channels=3, classes=1):
        super().__init__()

        # Encoder
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer2 = make_layer(in_channels=64, out_channels=64, num=3, stride=1)
        self.layer3 = make_layer(in_channels=64, out_channels=128, num=4, stride=2)
        self.layer4 = make_layer(in_channels=128, out_channels=256, num=6, stride=2)
        self.layer5 = make_layer(in_channels=256, out_channels=512, num=3, stride=2)

        # Decoder
        self.up = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            CBAM(256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        )

        self.layer7 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            CBAM(128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        )

        self.layer8 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            CBAM(64),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        )

        self.layer9 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=classes, kernel_size=1),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        def align_and_cat(up_tensor, skip_tensor):
            if up_tensor.shape[2:] != skip_tensor.shape[2:]:
                up_tensor = F.interpolate(up_tensor, size=skip_tensor.shape[2:], 
                                          mode='bilinear', align_corners=True)
            return torch.cat([up_tensor, skip_tensor], dim=1)

        # Decoder
        x6 = self.up(x5)
        x6 = align_and_cat(x6, x4)
        x6 = self.layer6(x6)

        x7 = align_and_cat(x6, x3)
        x7 = self.layer7(x7)

        x8 = align_and_cat(x7, x2)
        x8 = self.layer8(x8)

        x9 = align_and_cat(x8, x1)
        x9 = self.layer9(x9)

        out = F.interpolate(x9, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)

        return out