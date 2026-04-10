#! /usr/bin/env python3
import torch
import torch.nn as nn

class EgoLanesHead(nn.Module):
    def __init__(self):
        super(EgoLanesHead, self).__init__()

        # Standard
        self.GeLU = nn.GELU()

        # Segmentation Head - Output Layers
        self.decode_layer_6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.decode_layer_7 = nn.Conv2d(256, 128, 3, 1, 1)
        self.decode_layer_8 = nn.Conv2d(128, 64, 3, 1, 1)
        self.upsample_0 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.upsample_1 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.out_layer = nn.Conv2d(64, 3, 3, 1, 1)


    def forward(self, neck):

        # Prediction
        d6 = self.decode_layer_6(neck)
        d6 = self.GeLU(d6)
        d7 = self.decode_layer_7(d6)
        d7 = self.GeLU(d7)
        d8 = self.decode_layer_8(d7)
        d8 = self.GeLU(d8)
        u0 = self.upsample_0(d8)
        u0 = self.GeLU(u0)
        u1 = self.upsample_1(u0)
        u1 = self.GeLU(u1)
        output = self.out_layer(u1)

        return output