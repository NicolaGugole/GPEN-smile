'''
This is a simplified training code of GPEN. It achieves comparable performance as in the paper.

@Created by rosinality

@Modified by yangxy (yangtao9009@gmail.com)
'''
import argparse
import math
from telnetlib import KERMIT
import wandb
import os
import cv2
import glob
from tqdm import tqdm
from torchmetrics import PeakSignalNoiseRatio

import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils

import __init_paths
from training.data_loader.dataset_face import FaceDataset
from face_model.gpen_model_condition_whole_model_concat_embedding import FullGenerator, Discriminator

from training.loss.id_loss import IDLoss
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

from training import lpips


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def heatmap_loss(fake_img, real_img, input_img, loss, step, heatmap_blur=None):
    real_heat = (real_img - input_img)**2
    fake_heat = (fake_img - input_img)**2
    real_heat[real_heat < torch.mean(real_heat)] = 0.  # avoid working on face parts non-relevant
    fake_heat[fake_heat < torch.mean(fake_heat)] = 0.  # avoid working on face parts non-relevant
    if heatmap_blur is not None:
        real_heat = heatmap_blur(real_heat)
        fake_heat = heatmap_blur(fake_heat)
    if get_rank() == 0 and step % 50 == 0:
        sample = torch.cat((input_img, fake_img, real_img, real_heat, fake_heat), 0) 
        # image = wandb.Image(sample)
        # wandb.log({'training_images': image}, step=step)

    return loss(real_heat, fake_heat)


def g_nonsaturating_loss(fake_pred, loss_funcs=None, fake_img=None, real_img=None, input_img=None, step=0, heatmap_blur=None):
    smooth_l1_loss, id_loss, l2_loss = loss_funcs
    
    loss = F.softplus(-fake_pred).mean()
    loss_l1 = smooth_l1_loss(fake_img, real_img)
    loss_id, __, __ = id_loss(fake_img, real_img, input_img)
    heat_loss = heatmap_loss(fake_img, real_img, input_img, l2_loss, step, heatmap_blur)
    loss += 10*loss_l1 + 1.0*loss_id + 10*heat_loss  # ANALYZE L1 Loss relevance

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths

def validation(model, lpips_func, args, device, iter):
    data_in = [os.path.join(args.in_path,i)[:-1] for i in open(args.val_files) if 'jpg' in i] # remove '\n' at the end
    data_out = [os.path.join(args.out_path,i)[:-1] for i in open(args.val_files) if 'jpg' in i] # remove '\n' at the end
    lq_files = [i.split(' ')[0] for i in data_in] 
    hq_files = [i.split(' ')[0] for i in data_out]
    labels = [torch.LongTensor([int(i.split(' ')[1])]) for i in data_in]

    assert len(lq_files) == len(hq_files)

    print('Input Validation len:', len(lq_files))
    psnr = PeakSignalNoiseRatio().to(device)

    measured_lpips = 0
    measured_psnr = 0
    model.eval()
    for lq_f, hq_f, label in zip(lq_files, hq_files, labels):
        img_lq = cv2.imread(lq_f, cv2.IMREAD_COLOR)
        img_t = torch.from_numpy(img_lq).to(device).permute(2, 0, 1).unsqueeze(0)
        img_t = (img_t/255.-0.5)/0.5
        img_t = F.interpolate(img_t, (args.size, args.size))
        img_lq_t = torch.flip(img_t, [1])

        img_hq = cv2.imread(hq_f, cv2.IMREAD_COLOR)
        img_t = torch.from_numpy(img_hq).to(device).permute(2, 0, 1).unsqueeze(0)
        img_t = (img_t/255.-0.5)/0.5
        img_t = F.interpolate(img_t, (args.size, args.size))
        img_hq_t = torch.flip(img_t, [1])

        img_lq_t = img_lq_t.to(device)
        img_hq_t = img_hq_t.to(device)
        label = label.to(device)
        
        with torch.no_grad():
            img_out, __ = model(img_lq_t, iter, label=label)
        
            img_hq_lpips = lpips.im2tensor(lpips.load_image(hq_f)).to(device)
            img_hq_lpips = F.interpolate(img_hq_lpips, (args.size, args.size))
            measured_lpips += lpips_func.forward(img_out, img_hq_lpips)
            measured_psnr += psnr(img_out, img_hq_t)
    
    return measured_lpips.data/len(lq_files), measured_psnr/len(lq_files)


def train(args, loader, generator, discriminator, losses, g_optim, d_optim, g_ema, lpips_func, device, heatmap_blur=None):
    # if get_rank() == 0:
        # wandb.init(project='lifting-cgpen')
        # wandb.config.update(vars(args))
    loader = sample_data(loader)

    pbar = range(0, args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module

    else:
        g_module = generator
        d_module = discriminator
 
    accum = 0.5 ** (32 / (10 * 1000))

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print('Done!')
            break

        input_img, output_img, label = next(loader)  # degraded_img, real_img
        input_img = input_img.to(device)
        output_img = output_img.to(device)
        label = label.to(device)

        requires_grad(generator, False)
        requires_grad(discriminator, True)
        # print('AM HERE')
        generated_output_img, _ = generator(input_img, i, label=label)  # fake_img
        # print('OUTPUT SIZE:', generated_output_img.shape)
        d_generated_output_img = discriminator(generated_output_img)  # fake_pred

        d_output_img = discriminator(output_img)  # real_pred
        d_loss = d_logistic_loss(d_output_img, d_generated_output_img)

        loss_dict['d'] = d_loss
        loss_dict['real_score'] = d_output_img.mean()
        loss_dict['fake_score'] = d_generated_output_img.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            output_img.requires_grad = True
            d_generated_output_img = discriminator(output_img)
            r1_loss = d_r1_loss(d_generated_output_img, output_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * d_generated_output_img[0]).backward()

            d_optim.step()

        loss_dict['r1'] = r1_loss

        requires_grad(generator, True)
        requires_grad(discriminator, False)

        generated_output_img, _ = generator(input_img, i, label=label)
        d_generated_output_img = discriminator(generated_output_img)
        g_loss = g_nonsaturating_loss(d_generated_output_img, losses, fake_img=generated_output_img, real_img=output_img, input_img=input_img, step=i, heatmap_blur=heatmap_blur)

        loss_dict['g'] = g_loss

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        g_regularize = i % args.g_reg_every == 0

        if g_regularize:

            generated_output_img, latents = generator(input_img, i, return_latents=True, label=label)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                generated_output_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * generated_output_img[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()

        loss_dict['path'] = path_loss
        loss_dict['path_length'] = path_lengths.mean()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced['d'].mean().item()
        g_loss_val = loss_reduced['g'].mean().item()
        r1_val = loss_reduced['r1'].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f'd: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; '
                )
            )

            # if i % 10 == 0:
            #     wandb.log({
            #         'd_loss': d_loss_val,
            #         'g_loss': g_loss_val,
            #         'r1': r1_val
            #     }, step=i)
            
            if i % args.save_freq == 0:
                with torch.no_grad():
                    g_ema.eval()
                    sample, _ = g_ema(input_img, i, label=label)
                    sample = torch.cat((input_img, sample, output_img), 0) 
                    utils.save_image(
                        sample,
                        f'{args.sample}/{str(i).zfill(6)}.png',
                        nrow=args.batch,
                        normalize=True,
                        range=(-1, 1),
                    )
                    # image = wandb.Image(sample)
                    # wandb.log({'images': image}, step=i)

                lpips_value, psnr_value = validation(g_ema, lpips_func, args, device, i)
                print(f'VALIDATION --> {i}/{args.iter}: lpips: {lpips_value.cpu().numpy()[0][0][0][0]} - psnr: {psnr_value.cpu().item()}')
                # wandb.log({'lpips': lpips_value.cpu().numpy()[0][0][0][0]}, step=i)
                # wandb.log({'psnr': psnr_value.cpu().item()}, step=i)


            if i and i % args.save_freq == 0:
                torch.save(
                    {
                        'g': g_module.state_dict(),
                        'd': d_module.state_dict(),
                        'g_ema': g_ema.state_dict(),
                        'g_optim': g_optim.state_dict(),
                        'd_optim': d_optim.state_dict(),
                    },
                    f'{args.ckpt}/{str(i).zfill(6)}.pth',
                )
    # if get_rank() == 0:
    #     wandb.finish()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--in_path', type=str, default='/home/wizard/buckets/bsp-ai-science-public-datasets/stylegan-beautification-various-levels/input')
    parser.add_argument('--out_path', type=str, default='/home/wizard/buckets/bsp-ai-science-public-datasets/stylegan-beautification-various-levels/output')
    parser.add_argument('--train_files', type=str, default='/home/wizard/buckets/bsp-ai-science-public-datasets/stylegan-beautification-various-levels/train.txt')
    parser.add_argument('--val_files', type=str, default='/home/wizard/buckets/bsp-ai-science-public-datasets/stylegan-beautification-various-levels/val.txt')
    parser.add_argument('--base_dir', type=str, default='/home/wizard/buckets/bsp-ai-science-scratch/nicg/checkpoints/smilification')
    parser.add_argument('--iter', type=int, default=4000000)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--narrow', type=float, default=1.0)
    parser.add_argument('--r1', type=float, default=10)
    parser.add_argument('--path_regularize', type=float, default=2)
    parser.add_argument('--path_batch_shrink', type=int, default=2)
    parser.add_argument('--d_reg_every', type=int, default=16)
    parser.add_argument('--g_reg_every', type=int, default=4)
    parser.add_argument('--save_freq', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--sample', type=str, default='sample')
    parser.add_argument('--val_dir', type=str, default='val')

    args = parser.parse_args()

    os.makedirs(args.ckpt, exist_ok=True)
    os.makedirs(args.sample, exist_ok=True)

    device = 'cuda'

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0
    heatmap_blur = transforms.GaussianBlur(101)

    generator = FullGenerator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier, narrow=args.narrow, device=device
    ).to(device)

    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier, narrow=args.narrow, device=device
    ).to(device)
    g_ema = FullGenerator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier, narrow=args.narrow, device=device
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)
    
    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )

    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.pretrain is not None:
        print('load model:', args.pretrain)
        
        ckpt = torch.load(args.pretrain)

        generator.load_state_dict(ckpt['g'])
        discriminator.load_state_dict(ckpt['d'])
        g_ema.load_state_dict(ckpt['g_ema'])
            
        g_optim.load_state_dict(ckpt['g_optim'])
        d_optim.load_state_dict(ckpt['d_optim'])
    
    smooth_l1_loss = torch.nn.SmoothL1Loss().to(device)
    id_loss = IDLoss(args.base_dir, device, ckpt_dict=None)
    lpips_func = lpips.LPIPS(net='alex',version='0.1').to(device)
    l2_loss = torch.nn.MSELoss().to(device)
    
    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        id_loss = nn.parallel.DistributedDataParallel(
            id_loss,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    dataset = FaceDataset(args.in_path, args.out_path, args.train_files, args.size)
    print('Len of training data:', dataset.length)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )
    torch.autograd.set_detect_anomaly(True)

    train(args, loader, generator, discriminator, [smooth_l1_loss, id_loss, l2_loss], g_optim, d_optim, g_ema, lpips_func, device, heatmap_blur)
   
