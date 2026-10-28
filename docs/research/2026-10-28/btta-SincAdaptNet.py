import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math


class SincConv(nn.Module):
    """
    SincConv layer adapted from SincNet, which implements a learnable band-pass filter.
    This layer is particularly useful for processing signals such as EEG data by focusing on specific frequency bands.
    """

    def __init__(self, n_filt, filt_dim, fs):
        super(SincConv, self).__init__()
        if filt_dim % 2 == 0:
            filt_dim = filt_dim + 1
        self.n_filt = n_filt
        self.filt_dim = filt_dim
        self.fs = fs

        # Set minimum cutoff frequency and bandwidth
        self.min_freq = 0.5  # Allow filters to start from 0 Hz
        self.min_band = 4.0  # Set a small minimum bandwidth to ensure diversity

        # Initialize learnable parameters
        b1 = np.linspace(self.min_freq, 50.0, n_filt)  # Initialize lower cutoffs evenly across 0-100 Hz
        b2 = b1 + self.min_band  # Initialize with a small bandwidth

        filt_b1 = torch.from_numpy(b1 / self.fs).float()
        filt_band = torch.from_numpy((b2 - b1) / self.fs).float()
        self.filt_b1 = nn.Parameter(filt_b1)  # Lower cutoff frequency
        self.filt_band = nn.Parameter(filt_band)  # Bandwidth

        # Define the Hamming window function
        hamming_window = torch.from_numpy(np.hamming(filt_dim)).float()
        self.register_buffer('hamming_window', hamming_window)

        # Define the time points for the sinc function
        t_right = torch.linspace(1, filt_dim // 2, steps=filt_dim // 2) / fs
        self.register_buffer('t_right', t_right.float())

    def forward(self, x):
        # Convert bandwidth to upper cutoff and enforce >= minimum values
        filt_beg_freq = torch.abs(self.filt_b1) + self.min_freq / self.fs
        filt_end_freq = filt_beg_freq + (torch.abs(self.filt_band) + self.min_band / self.fs)
        filt_end_freq = torch.min(filt_end_freq, torch.tensor(100.0 / self.fs, device=filt_end_freq.device))

        # Construct the filter bank using the sinc function
        low_pass1 = 2 * filt_beg_freq[:, None] * self.sinc(filt_beg_freq * self.fs, self.t_right)
        low_pass2 = 2 * filt_end_freq[:, None] * self.sinc(filt_end_freq * self.fs, self.t_right)
        band_pass = (low_pass2 - low_pass1)

        max_band, _ = torch.max(band_pass, dim=1, keepdim=True)
        band_pass = band_pass / max_band  # Normalize the filter bank
        # filters = band_pass
        filters = band_pass * self.hamming_window[None, :]  # Apply Hamming window

        # Adjusting dimensions for conv2d
        filters = filters.view(self.n_filt, 1, 1, self.filt_dim)  # Convert filter to 4D for conv2d

        # Apply padding to preserve the length dimension and convolve
        padding = (0, (self.filt_dim - 1) // 2)
        out = F.conv2d(x, filters, padding=padding)

        return out

    def sinc(self, band, t_right):
        """
        Compute the sinc function used to generate the filter bank.

        :param band: Frequency band (n_filt,)
        :param t_right: Time points (filt_dim // 2,)
        :return: Sinc function applied to the filter bank (n_filt, filt_dim)
        """
        n_filt = band.size(0)
        band = band[:, None]  # Reshape for broadcasting (n_filt, 1)
        t_right = t_right[None, :]  # Reshape for broadcasting (1, filt_dim // 2)
        y_right = torch.sin(2 * math.pi * band * t_right) / (
                    2 * math.pi * band * t_right)  # Sinc function (n_filt, filt_dim // 2)
        y_left = torch.flip(y_right, [1])  # Mirror the right half
        y = torch.cat([y_left, torch.ones([n_filt, 1], device=band.device), y_right],
                      dim=1)  # Concatenate to form the full sinc filter (n_filt, filt_dim)
        return y


class SincAdaptNet(nn.Module):
    def __init__(self, input_channel, n_classes, n_spatial=16, n_fir=32, fir_length=51,
                 batch_norm_alpha=0.1, fs=250):
        super(SincAdaptNet, self).__init__()
        self.__dict__.update(locals())

        self.n_fir = n_fir

        # Spatial convolution
        self.conv_spatial = nn.Conv2d(1, self.n_spatial, (self.input_channel, 1), stride=(1, 1), bias=False)

        # Temporal convolution for filtering
        # self.conv_temporal = nn.Conv2d(1, self.n_fir, (1, self.fir_length), stride=(1, 1), bias=False, padding="same")
        # self.conv_temporal.weight.requires_grad = False
        self.conv_temporal = SincConv(self.n_fir, self.fir_length, fs=fs)

        # Layer normalization
        self.LN1 = nn.LayerNorm(24 * 16, eps=1e-5, )
        self.LN2 = nn.LayerNorm(32, eps=1e-5, )

        # Classification convolution layers
        self.conv_class1 = nn.Conv2d(self.n_fir, 32, kernel_size=(self.n_spatial, 1), stride=(1, 1), bias=False)
        self.conv_class4 = nn.Conv2d(1, self.n_classes, kernel_size=(32, 1), stride=(1, 1), bias=False)

        self.poolmean = nn.AvgPool2d(kernel_size=(1, 125))
        self.softmax = nn.LogSoftmax(dim=1)

        # Initialize filters
        self.initialize_filters()

    def initialize_filters(self):
        nn.init.xavier_uniform_(self.conv_spatial.weight, gain=1)
        # fil = np.zeros([self.n_fir, 1, 1, self.fir_length])
        # for i, lh in enumerate(self.fir_bank):
        #     fil[i, 0, 0, :] = create_bandpass_filter(fs=250, fl=lh[0], fh=lh[1], length=self.fir_length)
        # time_tensor = torch.Tensor(fil)
        # self.conv_temporal.weight = torch.nn.Parameter(time_tensor)
        # self.conv_temporal.weight.requires_grad = False
        nn.init.xavier_uniform_(self.conv_class1.weight, gain=1)
        nn.init.xavier_uniform_(self.conv_class4.weight, gain=1)

    def forward(self, x):
        x = self.conv_spatial(x)
        x1 = x.permute(0, 2, 1, 3)
        x2 = self.conv_temporal(x1)

        x2 = x2.reshape(-1, 24 * 16, x1.shape[-1])
        x2 = x2.permute(0, 2, 1)

        x = self.LN1(x2)
        x = x.permute(0, 2, 1).reshape(-1, 24, 16, x1.shape[-1])

        x = nn.functional.elu(self.conv_class1(x))
        x = x.permute(0, 2, 1, 3)

        x = x.reshape(-1, 1 * 32, x1.shape[-1]).permute(0, 2, 1)

        x = self.LN2(x)
        x = x.permute(0, 2, 1).reshape(-1, 1, 32, x1.shape[-1])

        x = self.conv_class4(x)
        x = self.softmax(x)
        x = x.squeeze(2)
        # x = torch.mean(x, -1)

        return x
