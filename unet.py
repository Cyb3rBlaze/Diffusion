import math

import torch
from torch import nn

import numpy as np

from config import Config


# pulled from Dr. Karpathy's minGPT implementation
class GELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


# Encoder block used to help create latent representation of pre-evolution pokemon
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, device=torch.device("cpu")):
        super().__init__()

        # can be compared to positional encodings in transformers
        self.time_linear = nn.Linear(time_dim, out_channels)

        # two convolution layers per encoder block
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding="same", device=device)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding="same", device=device)

        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.b_norm = nn.BatchNorm2d(out_channels, device=device)
        self.gelu = GELU()
    

    def forward(self, x, t):
        # processing time encodings
        t = self.time_linear(t)
        t = self.gelu(t)

        # expanding dims to the right for proper projection
        t = t[(..., ) + (None,)*2]

        # main operations
        out = self.conv1(x)
        out = self.b_norm(out)
        out = self.gelu(out)

        # element wise addition of time encodings
        out = out + t

        out = self.conv2(out)
        out = self.pool(out)
        out = self.b_norm(out)

        return self.gelu(out)


# Decoder block used to help construct evolved pokemon from latent representation
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, device=torch.device("cpu")):
        super().__init__()

        # can be compared to positional encodings in transformers
        self.time_linear = nn.Linear(time_dim, out_channels)

        # two convolution layers per decoder block, in_channels*2 because concatenating residuals
        self.conv1 = nn.Conv2d(in_channels*2, out_channels, kernel_size=(3, 3), padding="same", device=device)
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=(4, 4), stride=2, padding=1, device=device)
        
        self.b_norm = nn.BatchNorm2d(out_channels, device=device)
        self.gelu = GELU()
    

    def forward(self, x, t):
        # processing time encodings at every block level
        t = self.time_linear(t)
        t = self.gelu(t)

        # expanding dims for proper projection
        t = t[(..., ) + (None,)*2]

        # main operations
        out = self.conv1(x)
        out = self.b_norm(out)
        out = self.gelu(out)

        # element wise addition of time encodings
        out = out + t

        out = self.conv2(out)
        out = self.b_norm(out)

        return self.gelu(out)


# Same sinusoidal time/position embeddings as transformers
class SinusoidalTimeEmbeddings(nn.Module):
    def __init__(self, time_dim):
        super().__init__()
        self.time_dim = time_dim
    
    def forward(self, time_step_val):
        half_dim = self.time_dim//2
        embeddings = math.log(10000)/(half_dim-1)
        embeddings = torch.exp(torch.arange(half_dim) * -embeddings)
        embeddings = time_step_val[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


# VAE extracts latent features from pre-evolution image to produce evolution image
class UNet(nn.Module):
    def __init__(self, config, device=torch.device("cpu")):
        super().__init__()

        self.device = device

        ### TIME MLP

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbeddings(config.time_dim),
            nn.Linear(config.time_dim, config.time_dim),
            GELU()
        )

        ### ENCODER - extracts latent features from images

        # input: batch_size x 1 x 128 x 128, output: batch_size x 32 x 64 x 64
        self.encoder_blocks = [EncoderBlock(1, config.encoder_in_channels, config.time_dim, device)]
        # input: batch_size x 32 x 64 x 64, output: batch_size x 64 x 32 x 32
        self.encoder_blocks += [EncoderBlock(config.encoder_in_channels, config.encoder_in_channels*2, config.time_dim, device)]
        # input: batch_size x 64 x 32 x 32, output: batch_size x 128 x 16 x 16
        self.encoder_blocks += [EncoderBlock(config.encoder_in_channels*2, config.encoder_in_channels*4, config.time_dim, device)]
        # input: batch_size x 128 x 16 x 16, output: batch_size x 256 x 8 x 8
        self.encoder_blocks += [EncoderBlock(config.encoder_in_channels*4, config.encoder_in_channels*8, config.time_dim, device)]

        self.encoder_blocks = nn.ModuleList(self.encoder_blocks)


        ### DECODER - generates evolution images from latent representation

        # input: batch_size x 256 x 8 x 8, output: batch_size x 128 x 16 x 16
        self.decoder_blocks = [DecoderBlock(int(config.decoder_in_channels), int(config.decoder_in_channels/2), config.time_dim, device)]
        # input: batch_size x 128 x 16 x 16, output: batch_size x 64 x 32 x 32
        self.decoder_blocks += [DecoderBlock(int(config.decoder_in_channels/2), int(config.decoder_in_channels/4), config.time_dim, device)]
        # input: batch_size x 64 x 32 x 32, output: batch_size x 32 x 64 x 64
        self.decoder_blocks += [DecoderBlock(int(config.decoder_in_channels/4), int(config.decoder_in_channels/8), config.time_dim, device)]
        # input: batch_size x 32 x 64 x 64, output: batch_size x 1 x 128 x 128
        self.decoder_blocks += [DecoderBlock(int(config.decoder_in_channels/8), 1, config.time_dim, device)]

        self.decoder_blocks = nn.ModuleList(self.decoder_blocks)

        # output needs to be normalized between -1->1
        self.tanh = nn.Tanh()

    
    def forward(self, x, timestep):
        out = x
        residuals = []

        t = self.time_mlp(timestep)

        for i in range(len(self.encoder_blocks)):
            out = self.encoder_blocks[i](out, t)
            residuals += [out]

        # residual connections to feed info about pre-evolved pokemon to synthesize better output
        for i in range(len(self.decoder_blocks)):
            out = torch.cat((out, residuals[len(residuals)-i-1]), dim=1)
            out = self.decoder_blocks[i](out, t)
        
        return self.tanh(out)
