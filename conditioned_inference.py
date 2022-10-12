import torch
from face_model.gpen_model import FullGenerator
from PIL import Image
from torchvision import utils
from torch.nn import functional as F
import os
import time
import cv2

IMG_DIR = '/home/wizard/GPEN-smile/examples/imgs/'
SAVE_DIR = '/home/wizard/GPEN-smile/inference_results'

if __name__ == '__main__':
    for n, i in enumerate(os.listdir(IMG_DIR)):
        if '.jpg' in i or '.png' in i:
            start = time.time()
            model_state = torch.load('/home/wizard/buckets/bsp-ai-science-scratch/nicg/checkpoints/smilification/checkpoints/condition-input/067000.pth')
            model = FullGenerator(1024, 512, 8, device='cuda:0').to('cuda:0')
            model.load_state_dict(model_state['g_ema'])
            print('model loaded in ', time.time() - start)
            print(f'{n}/{len(os.listdir(IMG_DIR))}')
            img_path = os.path.join(IMG_DIR, i)
            start = time.time()
            input = cv2.imread(img_path, cv2.IMREAD_COLOR)
            input = torch.from_numpy(input).cuda().permute(2, 0, 1).unsqueeze(0)
            img_t = (input/255.-0.5)/0.5
            img_t = F.interpolate(img_t, (1024, 1024))
            img_t = torch.flip(img_t, [1])
            for label in [0,1,2]:
                output, _ = model(img_t, 13500, label=torch.LongTensor([label]).to('cuda:0'))
                print('inference complete in', time.time() - start)
                utils.save_image(
                                    output,
                                    os.path.join(SAVE_DIR, str(label)+'_conditioned_'+img_path.split('/')[-1]),
                                    nrow=1,
                                    normalize=True,
                                    range=(-1, 1),
                                )
                print('image saved')
            del input, img_t, output, model, model_state

