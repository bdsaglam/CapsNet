from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import conv_output_shape, prod


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
        self.num_routes = prod(self.capsule_output_shape)
        self.num_capsules = num_capsules
        self.output_shape = (self.num_routes, self.num_capsules)

    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), self.num_routes, self.num_capsules)
        return self.squash(u)

    def squash(self, input_tensor, epsilon=1e-7):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True) + epsilon
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class ObjectCaps(nn.Module):
    def __init__(self, input_shape, num_capsules, out_channels, num_iterations=3):
        super().__init__()
        assert num_iterations > 0

        self.num_routes = input_shape[0]
        self.in_channels = input_shape[1]
        self.num_capsules = num_capsules
        self.num_iterations = num_iterations
        self.output_shape = (num_capsules, out_channels, 1)

        self.W = nn.Parameter(torch.randn(1, self.num_routes, num_capsules, out_channels, self.in_channels))

    def forward(self, x):
        device = self.W.device

        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)

        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)

        b_ij = torch.zeros(1, self.num_routes, self.num_capsules, 1).to(device)

        v_j = None
        for iteration in range(self.num_iterations):
            c_ij = F.softmax(b_ij, dim=1)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)

            if iteration < self.num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)

        return v_j.squeeze(1)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class Decoder(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Decoder, self).__init__()
        self.num_objects = input_shape[0]
        self.output_shape = output_shape
        self.reconstruction_layers = nn.Sequential(
            nn.Linear(prod(input_shape), 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.output_shape[0] * self.output_shape[1] * self.output_shape[2]),
            nn.Sigmoid()
        )

    def forward(self, obj_vectors, labels=None):
        device = next(iter(self.parameters())).device
        batch_size = obj_vectors.size(0)

        logits = torch.sqrt((obj_vectors ** 2).sum(2))
        probs = F.softmax(logits, dim=1)
        _, best_indices = probs.max(dim=1)
        mask = F.one_hot(best_indices.squeeze(1).detach(), num_classes=self.num_objects).float().to(device)

        masked_obj_vectors = (obj_vectors * mask[:, :, None, None]).view(batch_size, -1)
        reconstructions = self.reconstruction_layers(masked_obj_vectors).view(batch_size, *self.output_shape)
        return reconstructions, mask


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

        self.decoder = Decoder(input_shape=self.object_capsules.output_shape,
                               output_shape=input_shape)

        self.mse_loss = nn.MSELoss()

    def forward(self, image):
        obj_vectors = self.object_capsules(self.primary_capsules(self.conv_layer(image)))
        reconstruction, masked = self.decoder(obj_vectors)
        return obj_vectors, reconstruction, masked

    def loss(self, batch_obj_vectors, batch_gt_label, batch_image, batch_reconstruction, reconstruction_loss_coef=5e-4):
        return self.margin_loss(batch_obj_vectors, batch_gt_label) \
               + reconstruction_loss_coef * self.reconstruction_loss(batch_image, batch_reconstruction)

    def margin_loss(self, batch_obj_vectors, batch_gt_label):
        num_classes = batch_obj_vectors.size(1)
        batch_label_embedding = F.one_hot(batch_gt_label, num_classes=num_classes).float()

        batch_size = batch_obj_vectors.size(0)

        v_c = torch.sqrt((batch_obj_vectors ** 2).sum(dim=2, keepdim=True))

        left = F.relu(-v_c + 0.9).view(batch_size, -1)
        right = F.relu(v_c - 0.1).view(batch_size, -1)

        loss = batch_label_embedding * left + 0.5 * (1.0 - batch_label_embedding) * right
        loss = loss.sum(dim=1).mean()

        return loss

    def reconstruction_loss(self, batch_image, batch_reconstruction):
        loss = self.mse_loss(batch_reconstruction.view(batch_reconstruction.size(0), -1),
                             batch_image.view(batch_reconstruction.size(0), -1))
        return loss
