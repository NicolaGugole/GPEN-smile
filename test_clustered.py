'''
@created by nicg
'''
import argparse
import os
import cv2
from torchmetrics import PeakSignalNoiseRatio
from deepface import DeepFace

import torch
from torch.nn import functional as F
from torch.utils import data

# from training.data_loader.dataset_face import FaceDataset
from face_model.gpen_model_classic import FullGenerator
from distributed import (
    synchronize,
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





def test(model_man, model_woman, lpips_func, args, device, iter):
    lq_files = [os.path.join(args.in_path,i)[:-1] for i in open(args.test_files) if 'jpg' in i]  # remove '\n' at the end
    hq_files = [os.path.join(args.out_path,i)[:-1] for i in open(args.test_files) if 'jpg' in i]

    assert len(lq_files) == len(hq_files)

    print('Input Test len:', len(lq_files))
    psnr = PeakSignalNoiseRatio().to(device)

    measured_lpips = 0
    measured_psnr = 0
    model_man.eval()
    model_woman.eval()
    for lq_f, hq_f in zip(lq_files, hq_files):
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
        img_gender = DeepFace.analyze(img_path = lq_f, actions = ['gender'], enforce_detection=False)['gender']
        if img_gender not in ['Man', 'Woman']:
            model = None
        else:
            model = model_man if img_gender == 'Man' else model_woman

        
        with torch.no_grad():
            img_out, __ = model(img_lq_t, iter)
        
            img_hq_lpips = lpips.im2tensor(lpips.load_image(hq_f)).to(device)
            img_hq_lpips = F.interpolate(img_hq_lpips, (args.size, args.size))
            measured_lpips += lpips_func.forward(img_out, img_hq_lpips)
            measured_psnr += psnr(img_out, img_hq_t)
    
    return measured_lpips.data/len(lq_files), measured_psnr/len(lq_files)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--in_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--test_files', type=str, required=True)
    parser.add_argument('--base_dir', type=str, default='/home/wizard/buckets/bsp-ai-science-scratch/nicg/checkpoints/smilification')
    parser.add_argument('--iter', type=int, default=4000000)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--narrow', type=float, default=1.0)
    parser.add_argument('--path_regularize', type=float, default=2)
    parser.add_argument('--path_batch_shrink', type=int, default=2)
    parser.add_argument('--save_freq', type=int, default=1000)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--man_ckpt', type=str, default='/home/wizard/buckets/bsp-ai-science-scratch/nicg/checkpoints/lifting-gpen/faceapp-remini/man_gpen/042000.pth')
    parser.add_argument('--woman_ckpt', type=str, default='/home/wizard/buckets/bsp-ai-science-scratch/nicg/checkpoints/lifting-gpen/faceapp-remini/woman_gpen/022000.pth')

    args = parser.parse_args()

    device = 'cuda:0'

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()

    args.latent = 512
    args.n_mlp = 8

    lpips_list = []
    psnrs_list = []
    args.start_iter = 0
    man_ckpt = args.man_ckpt
    woman_ckpt = args.woman_ckpt
    
    man_g_ema = FullGenerator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier, narrow=args.narrow, device=device
    ).to(device)
    man_g_ema.eval()
    print('load model:', man_ckpt)
    ckpt = torch.load(man_ckpt)
    man_g_ema.load_state_dict(ckpt['g_ema'])

    woman_g_ema = FullGenerator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier, narrow=args.narrow, device=device
    ).to(device)
    woman_g_ema.eval()
    print('load model:', woman_ckpt)
    ckpt = torch.load(woman_ckpt)
    woman_g_ema.load_state_dict(ckpt['g_ema'])

    lpips_func = lpips.LPIPS(net='alex',version='0.1').to(device)
    lpips_value, psnr_value = test(man_g_ema, woman_g_ema, lpips_func, args, device, 10000)
    print(f'Results with {woman_ckpt} as woman and {man_ckpt} - lpips_value: {lpips_value} - psnr_value: {psnr_value}')
    print('LPIPS:',lpips_value.cpu().squeeze().item())
    print('PSNR:',psnr_value.cpu().squeeze().item())
   
