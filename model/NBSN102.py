import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Conv2d_relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class N_BSN(nn.Module): # 1.02m version as described in the AT-BSN paper.
    def __init__(self):
        super(N_BSN, self).__init__()
        self.enc1 = Conv2d_relu(in_channels=3, out_channels =48, kernel_size =3, stride=1, padding =1)
        self.enc2 = Conv2d_relu(in_channels=48, out_channels =48, kernel_size =3, stride=1, padding =1)
        self.enc3 = Conv2d_relu(in_channels=48, out_channels =48, kernel_size =3, stride=1, padding =1)
        self.enc4 = Conv2d_relu(in_channels=48, out_channels =48, kernel_size =3, stride=1, padding =1)
        self.enc5 = Conv2d_relu(in_channels=48, out_channels =48, kernel_size =3, stride=1, padding =1)
        self.enc6 = Conv2d_relu(in_channels=48, out_channels =48, kernel_size =3, stride=1, padding =1)

        self.dec5_2 = Conv2d_relu(in_channels=96, out_channels=96, kernel_size =3, stride=1, padding =1)
        self.dec5_1 = Conv2d_relu(in_channels=96, out_channels=96, kernel_size =3, stride=1, padding =1)
        self.dec4_2 = Conv2d_relu(in_channels=144, out_channels=96, kernel_size =3, stride=1, padding =1)
        self.dec4_1 = Conv2d_relu(in_channels=96, out_channels=96, kernel_size =3, stride=1, padding =1)
        self.dec3_2 = Conv2d_relu(in_channels=144, out_channels=96, kernel_size =3, stride=1, padding =1)
        self.dec3_1 = Conv2d_relu(in_channels=96, out_channels=96, kernel_size =3, stride=1, padding =1)
        self.dec2_2 = Conv2d_relu(in_channels=144, out_channels=96, kernel_size =3, stride=1, padding =1)
        self.dec2_1 = Conv2d_relu(in_channels=96, out_channels=96, kernel_size =3, stride=1, padding =1)
        self.dec1_2 = Conv2d_relu(in_channels=144, out_channels=96, kernel_size =3, stride=1, padding =1)
        self.dec1_1 = nn.Conv2d(in_channels=96, out_channels=3, kernel_size =3, stride=1, padding =1)
        self.pool = nn.MaxPool2d(kernel_size = 2)
        self.unpool = nn.Upsample(scale_factor=2, mode="nearest")

        with torch.no_grad():
            self._init_weights()

    def forward(self, x):
        enc1 = self.enc1(x)
        pool1 = self.pool(enc1)
        enc2 = self.enc2(pool1)
        pool2 = self.pool(enc2)
        enc3 = self.enc3(pool2)
        pool3 = self.pool(enc3)
        enc4 = self.enc4(pool3)
        pool4 = self.pool(enc4)
        enc5 = self.enc5(pool4)
        pool5 = self.pool(enc5)
        enc6 = self.enc6(pool5)

        unpool5 = self.unpool(enc6)
        cat5 = torch.cat((unpool5, enc5),dim =1)
        dec5_2 = self.dec5_2(cat5)
        dec5_1 = self.dec5_1(dec5_2)
        unpool4 = self.unpool(dec5_1)
        cat4 = torch.cat((unpool4, enc4),dim =1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)
        unpool3 = self.unpool(dec4_1)
        cat3 = torch.cat((unpool3, enc3),dim =1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)
        unpool2 = self.unpool(dec3_1)
        cat2 = torch.cat((unpool2, enc2),dim =1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)
        unpool1 = self.unpool(dec2_1)
        cat1 = torch.cat((unpool1, enc1),dim =1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)
        return dec1_1

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0.1)
                m.bias.data.zero_()
