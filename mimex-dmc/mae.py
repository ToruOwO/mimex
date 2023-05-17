import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm.optim.optim_factory as optim_factory

import mae_models
from mae_transforms import random_resized_crop, horizontal_flip


class DataAug(nn.Module):
    '''RandomResizedCrop + RandomHorizontalFlip'''
    def __init__(
        self, out_size, scale=(0.8, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation='bicubic', flip_prob=0.5):
        super().__init__()
        self.out_size = out_size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.flip_prob = flip_prob

    def forward(self, x):
        """
        Args:
            x: (B * T, 2, 3, H, W) where T == action_repeat (frame stack)
        Returns:
            x_out: (B * T, 2, 3, out_size, out_size)
        """
        b, t, c, h, w = x.shape
        assert h == w

        x_out = torch.zeros(
            b, t, c, self.out_size, self.out_size, device=x.device)
        for bid in range(b):
            # random resized crop
            a = random_resized_crop(
                x[bid], self.out_size, self.out_size, self.scale, self.ratio,
                interpolation=self.interpolation)
            # random flip
            a = horizontal_flip(a, self.flip_prob)
            x_out[bid] = a

        return x_out


def get_data_aug(out_size):
    return transforms.Compose([
        transforms.RandomResizedCrop(
            out_size, scale=(0.8, 1.0), interpolation=3),  # bicubic
        transforms.RandomHorizontalFlip(),
    ])


def save_image(image, save_dir='/home/', fn='test'):
    # image is [H, W, C]
    assert image.shape[2] == 3
    im = torch.clip(image.detach().cpu() * 255, 0, 255).int().numpy().astype(
        np.uint8)
    plt.imshow(im)
    plt.title('', fontsize=16)
    plt.axis('off')
    plt.imsave(os.path.join(save_dir, f'{fn}.png'), im)


class MAE:
    def __init__(
        self, model_cfg, input_size=96, mask_ratio=0.9, num_mask_samples=10,
        weight_decay=0.05, base_lr=1e-3, save_vis=False, save_dir='',
        num_vis_samples=10, use_single_frame=False, no_frame_stack=False):
        self.mask_ratio = mask_ratio
        self.num_mask_samples = num_mask_samples
        self.num_vis_samples = num_vis_samples
        self.use_single_frame = use_single_frame
        self.no_frame_stack = no_frame_stack

        self.model = mae_models.mae_vit_mini_patch8(
            embed_dim=model_cfg.embed_dim,
            depth=model_cfg.depth,
            num_heads=model_cfg.num_heads,
            decoder_embed_dim=model_cfg.decoder_embed_dim,
            decoder_depth=model_cfg.decoder_depth,
            decoder_num_heads=model_cfg.decoder_num_heads).cuda()
        self.data_aug = get_data_aug(input_size)

        # following MAE protocol
        param_groups = optim_factory.add_weight_decay(self.model, weight_decay)
        self.optimizer = torch.optim.AdamW(
            param_groups, lr=base_lr, betas=(0.9, 0.95))

        self.save_vis = save_vis
        self.save_dir = save_dir

    def visualize(self, x, y, mask):
        """
        x: (N, C, H, W) / (N, C, t, H, W)
        y: (N, C, H, W) / (N, C, t, H, W)
        mask: (N, L)
        """
        x = x.detach().cpu()

        y = self.model.unpatchify(y)

        c = y.shape[1]

        if self.use_single_frame:
            y = torch.einsum('nchw->nhwc', y).detach().cpu()
            # visualize the mask (N, L) -> (N, H*W, p*p*3)
            mask = mask.detach()
            mask = mask.unsqueeze(-1).repeat(
                1, 1, self.model.patch_embed.patch_size[0]**2 *c)
            mask = self.model.unpatchify(mask)  # 1 is removing, 0 is keeping
            mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

            x = torch.einsum('nchw->nhwc', x)
        else:
            t = y.shape[2]
            y = torch.einsum('ncthw->nchtw', y).detach().cpu()

            # visualize the mask (N, L) -> (N, H*W, p*p*3*t)
            mask = mask.detach()
            mask = mask.unsqueeze(-1).repeat(
                1, 1, self.model.patch_embed.patch_size[0]**2 * c * t)
            mask = self.model.unpatchify(mask)  # 1 is removing, 0 is keeping
            mask = torch.einsum('ncthw->nchtw', mask).detach().cpu()
            x = torch.einsum('ncthw->nchtw', x)

            # concatenate frames along width 'nchtw->nch(tw)'
            y = y.flatten(start_dim=3, end_dim=4)
            mask = mask.flatten(start_dim=3, end_dim=4)
            x = x.flatten(start_dim=3, end_dim=4)

            y = torch.einsum('nchw->nhwc', y)
            mask = torch.einsum('nchw->nhwc', mask)
            x = torch.einsum('nchw->nhwc', x)

        # masked image
        im_masked = x * (1 - mask)

        # MAE reconstruction pasted with visible patches
        im_paste = x * (1 - mask) + y * mask

        for i in range(min(y.shape[0], self.num_vis_samples)):
            im_masked = x[i] * (1 - mask[i])
            im_paste = x[i] * (1 - mask[i]) + y[i] * mask[i]
            img = torch.cat((x[i], im_masked, y[i], im_paste))
            save_image(img, save_dir=self.save_dir, fn=f'{i}')

    def forward_model(self, x, vis=False):
        """
        x: (B * T * t, C, H, W) / (B * T, t, C, H, W)
        vis: whether to visualize
        """
        C, H, W = x.shape[-3:]
        if not self.use_single_frame:
            # (B * T, t, C, H, W) -> (B * T, C, t, H, W)
            x = x.permute(0, 2, 1, 3, 4)

        total_loss = torch.zeros(x.shape[0], device=x.device)
        mae_loss = 0.
        self.optimizer.zero_grad()

        # mask multiple times per input
        for _ in range(self.num_mask_samples):
            loss, y, mask = self.model.update(
                x, self.mask_ratio, keep_batch_loss=True)
            mae_loss += loss.sum()

            # detach loss and add to loss tracker
            total_loss += loss.detach()

        # optimizer update
        mae_loss.backward()
        self.optimizer.step()

        if vis:
            self.visualize(x, y, mask)

        # average over number of mask trials
        total_loss = total_loss / self.num_mask_samples
        return total_loss  # (B * T * t,) / (B * T,)

    def update(self, obs, next_obs, vis=False):
        """
        Args:
            obs, next_obs:
                (B, T * 3, H, W) where T == action_repeat
                (B, C, H, W) if self.no_frame_stack
            vis: whether to visualize
        Returns:
            loss: (B,)
        """
        # note: there's already temporal downsampling due to frame skip

        assert obs.shape == next_obs.shape
        B, N, H, W = obs.shape

        if self.no_frame_stack:
            obs = obs.view(B, 1, N, H, W)
            next_obs = next_obs.view(B, 1, N, H, W)
            x = torch.stack([obs, next_obs], dim=2)  # (B, 1, 2, N, H, W)
        else:
            assert obs.shape[1] % 3 == 0

            T = obs.shape[1] // 3  # no. of (o, o') pairs
            obs = obs.view(B, T, 3, H, W)
            next_obs = next_obs.view(B, T, 3, H, W)

            # transform images and resize to input size
            x = torch.stack([obs, next_obs], dim=2)  # (B, T, 2, 3, H, W)

        x = x.float() / 255

        if self.use_single_frame:
            # (B * T * t, C, H, W)
            x = x.view(-1, 3, H, W)
        else:
            if self.no_frame_stack:
                x = x.view(B, 2, N, H, W)
            else:
                # (B * T, t, C, H, W)
                x = x.view(B * T, 2, 3, H, W)

        # apply the same transform to images in the same trajectory
        x = self.data_aug(x)

        # (B * T * t,) for MAE
        # (B * T,) for space-time MAE
        loss = self.forward_model(x, vis).view(B, -1).mean(dim=-1, keepdim=True)

        return loss