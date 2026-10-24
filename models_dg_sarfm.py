import copy
import math
import os
from typing import Union
from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import Block, to_2tuple
from timm.models.layers import trunc_normal_
from util.pos_embed import get_2d_sincos_pos_embed
import numpy as np
import torch.nn.functional as F
from util.output_target import SAR_Hog
from torchvision import transforms


class MLPForContrast(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, bn_momentum=0.1, bn_eps=1e-5):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.bn = nn.BatchNorm1d(hidden_dim, eps=bn_eps, momentum=bn_momentum, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x, with_norm=False):
        x = self.l1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.l2(x)
        if with_norm:
            x = F.normalize(x, dim=-1, p=2)
            return x
        return x

class MultiLossFunction:
    def __init__(self, loss_type='l2', reduction='mean', beta=1.0 / 3, epsilon=1e-6):
        self.loss_type = loss_type
        self.reduction = reduction
        self.beta = beta
        self.epsilon = epsilon

    def __call__(self, pred, target):
        if self.loss_type == 'l1':
            return self.l1_loss(pred, target)
        elif self.loss_type == 'l2':
            return self.l2_loss(pred, target)
        elif self.loss_type == 'smooth_l1':
            return self.smooth_l1_loss(pred, target)
        elif self.loss_type == 'log_l1':
            return self.log_l1_loss(pred, target)
        elif self.loss_type == 'log_l2':
            return self.log_l2_loss(pred, target)
        elif self.loss_type == 'huber':
            return self.huber_loss(pred, target)
        else:
            raise ValueError(f"nonsupported loss type: {self.loss_type}")

    def l1_loss(self, pred, target):
        loss = torch.abs(pred - target)
        return self._reduce_loss(loss)

    def l2_loss(self, pred, target):
        loss = (pred - target) ** 2
        return self._reduce_loss(loss)

    def smooth_l1_loss(self, pred, target):
        n = torch.abs(pred - target)
        cond = n < self.beta
        loss = torch.where(cond, 0.5 * n ** 2 / self.beta, n - 0.5 * self.beta)
        return self._reduce_loss(loss)

    def huber_loss(self, pred, target, delta=1.0):
        diff = torch.abs(pred - target)
        loss = torch.where(diff < delta, 0.5 * diff ** 2, delta * (diff - 0.5 * delta))
        return self._reduce_loss(loss)

    def log_l1_loss(self, pred, target):
        min_val = torch.min(torch.min(pred), torch.min(target))
        max_val = torch.max(torch.max(pred), torch.max(target))
        range_val = max_val - min_val

        if range_val < 1e-8:
            range_val = 1.0

        pred_pos = (pred - min_val) / range_val + self.epsilon
        target_pos = (target - min_val) / range_val + self.epsilon

        pred_log = torch.log(pred_pos)
        target_log = torch.log(target_pos)
        loss = torch.abs(pred_log - target_log)

        return self._reduce_loss(loss)

    def log_l2_loss(self, pred, target):
        min_val = torch.min(torch.min(pred), torch.min(target))
        max_val = torch.max(torch.max(pred), torch.max(target))
        range_val = max_val - min_val

        if range_val < 1e-8:
            range_val = 1.0

        pred_pos = (pred - min_val) / range_val + self.epsilon
        target_pos = (target - min_val) / range_val + self.epsilon

        pred_log = torch.log(pred_pos)
        target_log = torch.log(target_pos)
        loss = (pred_log - target_log) ** 2

        return self._reduce_loss(loss)

    def _reduce_loss(self, loss):
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"nonsupported reduction type: {self.reduction}")


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Clencoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm, mask_ratio=0.75,
                 init_cfg=None):
        super().__init__()
        self.init_cfg = init_cfg
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.mask_ratio = mask_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        # self.repr = nn.Linear(embed_dim, representation_size)
        # self.repr_act = nn.Tanh()
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        if self.init_cfg is None:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                                cls_token=True)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

            # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
            w = self.patch_embed.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

            # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
            torch.nn.init.normal_(self.cls_token, std=.02)

            # initialize nn.Linear and nn.LayerNorm
            self.apply(self._init_weights)

        else:
            assert os.path.exists(self.init_cfg), f"{self.init_cfg} file doesn't exist"
            print(f'loading initial weights from {self.init_cfg}')
            checkpoint = torch.load(self.init_cfg, map_location='cpu')
            checkpoint_model = checkpoint['model']
            msg = self.load_state_dict(checkpoint_model, strict=False)
            print(msg)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        if self.mask_ratio > 0:
            len_keep = int(L * (1 - self.mask_ratio))

            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

            # sort noise for each sample
            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=1)

            # keep the first subset
            ids_keep = ids_shuffle[:, :len_keep]
            x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

            # generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([N, L], device=x.device)
            mask[:, :len_keep] = 0
            # unshuffle to get the binary mask
            mask = torch.gather(mask, dim=1, index=ids_restore)

            return x_masked, mask, ids_restore
        else:
            ids_keep = torch.arange(0, L, device=x.device, dtype=torch.long)
            ids_keep = ids_keep.unsqueeze(dim=0).repeat(N, 1)
            x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

            ids_restore = torch.argsort(ids_keep, dim=1)
            mask = torch.zeros([N, L], device=x.device)

            return x_masked, mask, ids_restore

    def _resize_pos_embed(self, pos_embed, new_num_patches):
        """Resize pos_embed (1, old_n+1, D) -> (1, new_n+1, D) by bicubic interp on spatial grid."""
        if pos_embed is None:
            return None
        cls_tok = pos_embed[:, :1, :]  # (1,1,D)
        grid_tok = pos_embed[:, 1:, :]  # (1, old_n, D)
        old_n = int(math.sqrt(grid_tok.shape[1]))
        new_n = int(math.sqrt(new_num_patches))

        assert old_n * old_n == grid_tok.shape[1] and new_n * new_n == new_num_patches

        if old_n == new_n:
            return pos_embed
        D = grid_tok.shape[2]
        grid_tok = grid_tok.reshape(1, old_n, old_n, D).permute(0, 3, 1, 2)  # (1, D, H, W)
        grid_tok = F.interpolate(grid_tok, size=(new_n, new_n), mode='bicubic', align_corners=False)
        grid_tok = grid_tok.permute(0, 2, 3, 1).reshape(1, new_n * new_n, D)
        return torch.cat([cls_tok, grid_tok], dim=1)  # (1, new_n*new_n+1, D)

    def forward_no_mask(self, x: torch.Tensor):
        B = x.shape[0]
        x = self.patch_embed(x)
        num_patches_curr = x.shape[1]
        if self.pos_embed is not None and self.pos_embed.shape[1] != num_patches_curr + 1:
            pos_embed = self._resize_pos_embed(self.pos_embed, num_patches_curr)
        else:
            pos_embed = self.pos_embed

        x = x + pos_embed[:, 1:, :]

        cls_tokens = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_tokens.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        num_patches_curr = x.shape[1]
        if self.pos_embed is not None and self.pos_embed.shape[1] != num_patches_curr + 1:
            pos_embed = self._resize_pos_embed(self.pos_embed, num_patches_curr)
        else:
            pos_embed = self.pos_embed

        x = x + pos_embed[:, 1:, :]

        x, mask, id_restore = self.random_masking(x)

        cls_tokens = self.cls_token + pos_embed[:, :1, :]  # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = cls_tokens.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x, mask, id_restore


class ClDecoder(nn.Module):
    def __init__(self, embed_dim=768, num_patches=196, patch_size=16, in_chans=3,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, use_finalnorm=False, norm_layer=nn.LayerNorm):
        super().__init__()
        self.use_finalnorm = use_finalnorm
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.num_patches = num_patches

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)

        if self.use_finalnorm:
            self.final_norm = norm_layer(patch_size ** 2 * in_chans)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _resize_decoder_pos_embed(self, decoder_pos_embed, new_num_patches):
        """Resize decoder_pos_embed (1, old_n+1, D) -> (1, new_n+1, D) by bicubic interp on spatial grid."""
        if decoder_pos_embed is None:
            return None
        cls_tok = decoder_pos_embed[:, :1, :]  # (1,1,D)
        grid_tok = decoder_pos_embed[:, 1:, :]  # (1, old_n, D)
        old_n = int(math.sqrt(grid_tok.shape[1]))
        new_n = int(math.sqrt(new_num_patches))

        assert old_n * old_n == grid_tok.shape[1] and new_n * new_n == new_num_patches

        if old_n == new_n:
            return decoder_pos_embed
        D = grid_tok.shape[2]
        grid_tok = grid_tok.reshape(1, old_n, old_n, D).permute(0, 3, 1, 2)  # (1, D, H, W)
        grid_tok = F.interpolate(grid_tok, size=(new_n, new_n), mode='bicubic', align_corners=False)
        grid_tok = grid_tok.permute(0, 2, 3, 1).reshape(1, new_n * new_n, D)
        return torch.cat([cls_tok, grid_tok], dim=1)  # (1, new_n*new_n+1, D)

    def forward(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        num_patches_curr = x.shape[1] - 1
        if self.decoder_pos_embed is not None and self.decoder_pos_embed.shape[1] != num_patches_curr + 1:
            decoder_pos_embed = self._resize_decoder_pos_embed(self.decoder_pos_embed, num_patches_curr)
        else:
            decoder_pos_embed = self.decoder_pos_embed

        x = x + decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        if self.use_finalnorm:
            x = self.final_norm(x)
        return x


def _init_weights(m):
    if isinstance(m, nn.Linear):
        # we use xavier_uniform following official JAX ViT:
        torch.nn.init.xavier_uniform_(m.weight)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class DG_SARFM(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, cl_dim=256, mlp_dim=4096,
                 mask_ratio=0.75, cl_decoder_depth=2, init_cfg=None, momentum=0.996):
        super().__init__()

        # --------------------------------------------------------------------------
        # encoder (no momentum encoder)
        self.m = momentum
        self.encoder = Clencoder(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                                 depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, qk_scale=None,
                                 norm_layer=norm_layer, mask_ratio=mask_ratio, init_cfg=init_cfg)
        self.momentum_encoder = Clencoder(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                          embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                          qkv_bias=True, qk_scale=None, norm_layer=norm_layer, mask_ratio=0)

        self.num_patches = self.encoder.num_patches

        # projection head
        self.proj = MLPForContrast(embed_dim, mlp_dim, cl_dim)
        self.momentum_proj = MLPForContrast(embed_dim, mlp_dim, cl_dim)
        # self.pred = MLPForContrast(cl_dim, mlp_dim, cl_dim)

        # decoders
        self.rc_decoder = ClDecoder(embed_dim=embed_dim, num_patches=self.num_patches, patch_size=patch_size,
                                    in_chans=in_chans, decoder_embed_dim=decoder_embed_dim, decoder_depth=decoder_depth,
                                    decoder_num_heads=decoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True,
                                    qk_scale=None, use_finalnorm=False, norm_layer=norm_layer)
        self.cl_decoder = ClDecoder(embed_dim=embed_dim, num_patches=self.num_patches, patch_size=patch_size,
                                    in_chans=3, decoder_embed_dim=decoder_embed_dim, decoder_depth=cl_decoder_depth,
                                    decoder_num_heads=decoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True,
                                    qk_scale=None, use_finalnorm=True, norm_layer=norm_layer)

        _init_weights(self.proj)
        # _init_weights(self.pred)

        self.patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss

        self.output_target1 = SAR_Hog(kernel_size=9, p=4)
        self.output_target2 = SAR_Hog(kernel_size=13, p=4)
        self.output_target3 = SAR_Hog(kernel_size=17, p=4)

        for param_b, param_m in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)
            param_m.requires_grad = False

        for param_b, param_m in zip(self.proj.parameters(), self.momentum_proj.parameters()):
            param_m.data.copy_(param_b.data)
            param_m.requires_grad = False

    def _update_moment_encoder(self, m):
        for param_b, param_m in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1 - m)

        for param_b, param_m in zip(self.proj.parameters(), self.momentum_proj.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1 - m)

    def extract_feat(self, img):
        x = self.encoder(img)
        return x


    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward_rcloss(self, output_img, pred_pixel, mask):
        B, C, H, W = output_img.shape
        if C == 1:
            transform = transforms.Normalize(mean=[0.5], std=[0.5])
            target = self.patchify(transform(torch.cat([self.output_target1(output_img), self.output_target2(output_img), self.output_target3(output_img)], dim=1)))
        elif C == 3:
            target = self.patchify(output_img)
        else:
            raise ValueError('expect output to be 4 dims')
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        # loss = (pred_pixel - target) ** 2
        # loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss_fn = MultiLossFunction(loss_type='smooth_l1', beta=0.5, reduction='none')
        loss = loss_fn(pred=pred_pixel, target=target)

        if loss.dim() > 2:
            loss = loss.mean(dim=-1)
        

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_cl_loss(self, proj_q, proj_k, weight=None):
        # q = self.predictor(proj_q)
        loss = 2 - 2 * (proj_q * proj_k).sum(dim=-1)
        if weight is not None:
            loss = loss * weight
        loss = loss.mean()
        return loss

    def forward_train(self, input_img, output_img, contrast_img, ious=None, m=0.987):
        if not isinstance(input_img, (list, tuple)):
            input_img = [input_img]
        else:
            input_img = list(input_img)
        # assert len(input_img) == len(contrast_img), "noisy views and denoised views must match in number"

        # reconstruction loss from noisy view 0
        latent_q0, mask_q0, ids_restore_q0 = self.encoder(input_img[0])
        pred_pixel = self.rc_decoder(latent_q0, ids_restore_q0)
        rc_loss = self.forward_rcloss(output_img, pred_pixel, mask_q0)

        # compute target projections from the denoised view
        with torch.no_grad():
            self._update_moment_encoder(m)
            pred_feat_d0, mask_d0, ids_restore_d0 = self.momentum_encoder(contrast_img)
            proj_d0 = self.momentum_proj(torch.mean(pred_feat_d0[:, 1:, :], dim=1), with_norm=True)  # B x D
            # proj_d0 = self.proj(pred_feat_d0[:, 0, :], with_norm=True)

        # compute loss: for all the views
        loss_multiview = 0.0
        n_views = 0
        for v, noisy_view in enumerate(input_img):
            if v == 0:
                pred_feature = self.cl_decoder(latent_q0, ids_restore_q0)
                proj_q = self.proj(torch.mean(pred_feature, dim=1), with_norm=True)
                # proj_q = self.pred(proj_q, with_norm=True)
                if ious is not None:
                    iou1 = ious[:, 0]
                else:
                    iou1 = None
                loss_singleview = self.forward_cl_loss(proj_q=proj_q, proj_k=proj_d0.detach(), weight=iou1)
                continue

            latent_q, _, ids_restore_q = self.encoder(noisy_view)
            pred_feature = self.cl_decoder(latent_q, ids_restore_q)
            proj_q = self.proj(torch.mean(pred_feature, dim=1), with_norm=True)  # B x D

            if ious is not None:
                iou2 = ious[:, v]
            else:
                iou2 = None
            loss_v = self.forward_cl_loss(proj_q=proj_q, proj_k=proj_d0.detach(), weight=iou2)
            loss_multiview += loss_v
            if ious is None: 
                n_views += 1

        if ious is None and n_views > 0:
            loss_multiview = loss_multiview / float(n_views)

        loss = rc_loss + loss_multiview + loss_singleview
        return loss

    def forward_test(self, input_img):
        latent, mask, ids_restore = self.encoder(input_img)
        pred = self.rc_decoder(latent, ids_restore)

        pred = self.unpatchify(pred)
        pred = torch.einsum('nchw->nhwc', pred).detach().cpu()

        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, self.patch_size ** 2)  # (N, H*W, p*p*3)
        mask = self.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

        return mask, pred

    def forward(self, input_img, mode, **kwargs):
        if mode == 'tensor':
            feats = self.extract_feat(input_img)
            return feats
        elif mode == 'loss':
            return self.forward_train(input_img, **kwargs)
        elif mode == 'predict':
            return self.forward_test(input_img)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')


def sarfm_base(**kwargs):
    model = DG_SARFM(
        patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), cl_dim=128, mlp_dim=2048, cl_decoder_depth=1, **kwargs
    )
    return model

