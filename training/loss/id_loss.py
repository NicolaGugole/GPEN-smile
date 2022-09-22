import os
import torch
from torch import nn
from model_irse import Backbone

class IDLoss(nn.Module):
    def __init__(self, base_dir='./', device='cuda', ckpt_dict=None):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se').to(device)
        if ckpt_dict is None:
            self.facenet.load_state_dict(torch.load(os.path.join(base_dir, 'weights', 'model_ir_se50.pth'), map_location=torch.device('cpu')))
        else:
            self.facenet.load_state_dict(ckpt_dict)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

    def extract_feats(self, x):
        _, _, h, w = x.shape
        assert h==w
        ss = h//256
        x = x[:, :, 35*ss:-33*ss, 32*ss:-36*ss]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y, x):
        n_samples = x.shape[0]
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            loss += 1 - diff_target
            count += 1

        return loss / count, None, None

