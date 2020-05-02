from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .utils import conv_output_shape

USE_CUDA = True if torch.cuda.is_available() else False


class ConvLayer(nn.Module):
    def __init__(self, input_shape, out_channels, kernel_size, stride):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=input_shape[0],
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride)

        self.output_shape = conv_output_shape(input_shape=input_shape,
                                              out_channels=out_channels,
                                              kernel_size=kernel_size,
                                              stride=stride)

    def forward(self, x):
        return F.relu(self.conv(x))


class PrimaryCaps(nn.Module):
    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 num_capsules: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 2,
                 padding: int = 0):
        super().__init__()
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=input_shape[0],
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding)
            for _ in range(num_capsules)
        ])
        self.capsule_output_shape = conv_output_shape(input_shape=input_shape,
                                                      out_channels=out_channels,
                                                      kernel_size=kernel_size,
                                                      stride=stride,
                                                      padding=padding)
        self.num_routes = self.capsule_output_shape[0] * self.capsule_output_shape[1] * self.capsule_output_shape[2]
        self.num_capsules = num_capsules
        self.output_shape = (self.num_routes, self.num_capsules)

    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), self.num_routes, self.num_capsules)
        return self.squash(u)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class ObjectCaps(nn.Module):
    def __init__(self, input_shape, num_capsules, out_channels):
        super().__init__()

        self.num_routes = input_shape[0]
        self.in_channels = input_shape[1]
        self.num_capsules = num_capsules

        self.W = nn.Parameter(torch.randn(1, self.num_routes, num_capsules, out_channels, self.in_channels))

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)

        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)

        b_ij = Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1))
        if USE_CUDA:
            b_ij = b_ij.cuda()

        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, dim=1)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)

            if iteration < num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)

        return v_j.squeeze(1)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class Decoder(nn.Module):
    def __init__(self, output_shape):
        super(Decoder, self).__init__()
        self.output_shape = output_shape
        self.reconstruction_layers = nn.Sequential(
            nn.Linear(16 * 10, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.output_shape[0] * self.output_shape[1] * self.output_shape[2]),
            nn.Sigmoid()
        )

    def forward(self, x, data):
        classes = torch.sqrt((x ** 2).sum(2))
        classes = F.softmax(classes, dim=0)

        _, max_length_indices = classes.max(dim=1)
        masked = Variable(torch.sparse.torch.eye(10))
        if USE_CUDA:
            masked = masked.cuda()
        masked = masked.index_select(dim=0, index=Variable(max_length_indices.squeeze(1).data))
        t = (x * masked[:, :, None, None]).view(x.size(0), -1)
        reconstructions = self.reconstruction_layers(t)
        reconstructions = reconstructions.view(-1, *self.output_shape)
        return reconstructions, masked


class CapsNet(nn.Module):
    def __init__(self,
                 input_shape,
                 cnn_out_channels=256,
                 cnn_kernel_size=9,
                 cnn_stride=1,
                 pc_num_capsules=8,
                 pc_out_channels=32,
                 pc_kernel_size=9,
                 pc_stride=2,
                 obj_num_capsules=10,
                 obj_out_channels=16,
                 ):
        super(CapsNet, self).__init__()
        self.conv_layer = ConvLayer(input_shape=input_shape,
                                    out_channels=cnn_out_channels,
                                    kernel_size=cnn_kernel_size,
                                    stride=cnn_stride)

        self.primary_capsules = PrimaryCaps(input_shape=self.conv_layer.output_shape,
                                            num_capsules=pc_num_capsules,
                                            out_channels=pc_out_channels,
                                            kernel_size=pc_kernel_size,
                                            stride=pc_stride)

        self.object_capsules = ObjectCaps(input_shape=self.primary_capsules.output_shape,
                                          num_capsules=obj_num_capsules,
                                          out_channels=obj_out_channels)
        self.decoder = Decoder(output_shape=input_shape)

        self.mse_loss = nn.MSELoss()

    def forward(self, data):
        output = self.object_capsules(self.primary_capsules(self.conv_layer(data)))
        reconstructions, masked = self.decoder(output, data)
        return output, reconstructions, masked

    def loss(self, batch_image, x, target, reconstructions):
        return self.margin_loss(x, target) + self.reconstruction_loss(batch_image, reconstructions)

    def margin_loss(self, x, labels, size_average=True):
        batch_size = x.size(0)

        v_c = torch.sqrt((x ** 2).sum(dim=2, keepdim=True))

        left = F.relu(0.9 - v_c).view(batch_size, -1)
        right = F.relu(v_c - 0.1).view(batch_size, -1)

        loss = labels * left + 0.5 * (1.0 - labels) * right
        loss = loss.sum(dim=1).mean()

        return loss

    def reconstruction_loss(self, batch_image, reconstructions):
        loss = self.mse_loss(reconstructions.view(reconstructions.size(0), -1), batch_image.view(reconstructions.size(0), -1))
        return loss * 0.0005
