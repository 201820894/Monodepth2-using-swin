# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *


class Swin_DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(Swin_DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        #  Resnet encoder [64, 64, 128, 256, 512]

        #  Swin encoder
        #([4, 96, 48, 160])
        #([4, 192, 24, 80])
        #([4, 384, 12, 40])
        #([4, 768, 6, 20])

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 1:
                num_ch_in += self.num_ch_enc[i - 2]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(
                self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        """
        Resnet encoder 
        (8, 64, 96, 320)
        (8, 64, 48, 160)
        (8, 128, 24, 80)
        (8, 256, 12, 40)
        (8, 512, 6, 20)
        """

        """
        Swin encoder
        (8, 96, 48, 160)
        (8, 192, 24, 80)
        (8, 384, 12, 40)
        (8, 768, 6, 20)        
        """

        """
        Decoder channel
        {16, 32, 64, 128, 256}
        """

        x = input_features[-1]
        for i in range(4, -1, -1):
            # 3x3 conv 이후 ELU
            # (8 768, 6, 20)
            x = self.convs[("upconv", i, 0)](x)
            # (8, 256, 6, 20)

            x = [upsample(x)]
            # (8, 256, 12, 40)
            if self.use_skips and i > 1:
                x += [input_features[i - 2]]

            x = torch.cat(x, 1)
            # 여기까지 (8, 640, 12, 40)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(
                    self.convs[("dispconv", i)](x))


        return self.outputs
