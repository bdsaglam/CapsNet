from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import conv_output_shape, prod, squash, safe_norm


class ConvEncoder(nn.Module):
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
        u = [capsule(x) for capsule in self.capsules]  # [(B, C, G, G)] * P
        u = torch.stack(u, dim=1)  # (B, P, C, G, G)
        u = u.view(x.size(0), self.num_routes, self.num_capsules)  # (B, C*G*G, P)
        return squash(u, dim=-1)  # (B, C*G*G, P)


class ObjectCaps(nn.Module):
    def __init__(self, input_shape, num_capsules, out_channels,
                 num_iterations=3):
        super().__init__()
        assert num_iterations > 0

        self.num_routes = input_shape[0]
        self.in_channels = input_shape[1]
        self.num_capsules = num_capsules
        self.num_iterations = num_iterations
        self.output_shape = (num_capsules, out_channels, 1)

        self.W = nn.Parameter(
            torch.randn(1, self.num_routes, num_capsules, out_channels, self.in_channels),
            requires_grad=True
        )

    def forward(self, x):  # (B, C*G*G, P)
        device = self.W.device

        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)  # (B, C*G*G, O, P, 1)

        W = torch.cat([self.W] * batch_size, dim=0)  # (B, C*G*G, O, F, P)
        u_hat = torch.matmul(W, x)  # (B, C*G*G, O, F, 1)

        b_ij = torch.zeros(batch_size, self.num_routes, self.num_capsules, 1, 1).to(device)  # (B, C*G*G, O, 1, 1)

        v_j = None
        for iteration in range(self.num_iterations):
            c_ij = F.softmax(b_ij, dim=2)  # (B, C*G*G, O, 1, 1)
            # c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)  # (B, C*G*G, O, 1, 1)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)  # (B, 1, O, F, 1)
            v_j = squash(s_j, dim=3)  # (B, 1, O, F, 1)

            # update vote weights if not last iteration
            if iteration < self.num_iterations - 1:
                v_ij = torch.cat([v_j] * self.num_routes, dim=1)  # (B, C*G*G, O, F, 1)
                a_ij = torch.matmul(u_hat.transpose(3, 4), v_ij)  # (B, C*G*G, O, 1, 1)
                b_ij = b_ij + a_ij

        return v_j.squeeze(1)  # (B, O, F, 1)


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
            nn.Linear(1024, prod(output_shape)),
            nn.Sigmoid()
        )

    def forward(self, obj_vectors, label):
        device = next(iter(self.parameters())).device
        batch_size = obj_vectors.size(0)

        mask = F.one_hot(label, num_classes=self.num_objects).float().to(device)
        masked_obj_vectors = (obj_vectors * mask[:, :, None, None]).view(batch_size, -1)
        reconstruction = self.reconstruction_layers(masked_obj_vectors).view(batch_size, *self.output_shape)
        return reconstruction


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
        self.encoder = ConvEncoder(input_shape=input_shape,
                                   out_channels=cnn_out_channels,
                                   kernel_size=cnn_kernel_size,
                                   stride=cnn_stride)

        self.primary_capsules = PrimaryCaps(
            input_shape=self.encoder.output_shape,
            num_capsules=pc_num_capsules,
            out_channels=pc_out_channels,
            kernel_size=pc_kernel_size,
            stride=pc_stride
        )

        self.object_capsules = ObjectCaps(
            input_shape=self.primary_capsules.output_shape,
            num_capsules=obj_num_capsules,
            out_channels=obj_out_channels
        )

        self.decoder = Decoder(input_shape=self.object_capsules.output_shape, output_shape=input_shape)

    def forward(self, image, label=None):
        obj_vectors = self.object_capsules(self.primary_capsules(self.encoder(image)))  # (B, O, F, 1)

        y_prob = safe_norm(obj_vectors, dim=2, keepdim=False).squeeze(-1)  # (B, O)

        if label is None:
            _, best_indices = y_prob.max(dim=1)
            label = best_indices.detach()

        reconstruction = self.decoder(obj_vectors, label=label)
        return obj_vectors, reconstruction, y_prob

    def loss(self, obj_vectors, gt_label, image, reconstruction, reconstruction_loss_coef=5e-4):
        return self.margin_loss(obj_vectors, gt_label) \
               + reconstruction_loss_coef * self.reconstruction_loss(image, reconstruction)

    def margin_loss(self, y_prob, gt_label, m_plus=0.9, m_minus=0.1, lmd=0.5):
        batch_size, num_objects = y_prob.shape

        label_embedding = F.one_hot(gt_label, num_classes=num_objects).float()  # (B, O)

        over_suppressed = (F.relu(-y_prob + m_plus) ** 2).view(batch_size, -1)
        under_suppressed = (F.relu(y_prob - m_minus) ** 2).view(batch_size, -1)

        loss = label_embedding * over_suppressed + lmd * (1.0 - label_embedding) * under_suppressed
        loss = loss.sum(dim=1).mean()

        return loss

    def reconstruction_loss(self, image, reconstruction):
        batch_size = reconstruction.shape[0]
        loss = F.mse_loss(
            reconstruction.view(batch_size, -1),
            image.view(batch_size, -1)
        )
        return loss
