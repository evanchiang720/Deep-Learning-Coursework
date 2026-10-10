import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, channels=3, classes=1):
        super().__init__()

        def crop_and_cat(upsampled, bypass):
            diff = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass_cropped = bypass[:, :, diff:diff + upsampled.size()[2], diff:diff + upsampled.size()[3]]
            return torch.cat([bypass_cropped, upsampled], dim=1)
        self.crop_and_cat = crop_and_cat

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=0),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=0),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=0),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        )
        self.layer9 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=classes, kernel_size=1),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # Encoder
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        up4 = self.layer5(x4)
        
        x6_in = self.crop_and_cat(up4, x4)
        up3 = self.layer6(x6_in)
        
        x7_in = self.crop_and_cat(up3, x3)
        up2 = self.layer7(x7_in)
        
        x8_in = self.crop_and_cat(up2, x2)
        up1 = self.layer8(x8_in)
        
        x9_in = self.crop_and_cat(up1, x1)
        x9 = self.layer9(x9_in)

        return x9