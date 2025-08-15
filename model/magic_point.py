import numpy as np
#-*-coding:utf8-*-
import torch
from solver.nms import box_nms
from model.modules.cnn.vgg_backbone import VGGBackboneBN,VGGBackbone
from model.modules.cnn.cnn_heads import DetectorHead



class MagicPoint(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """

    def __init__(self, config, input_channel=1, grid_size=8, using_bn=True, device='cpu'):
        super(MagicPoint, self).__init__()
        self.nms = config['nms']
        self.det_thresh = config['det_thresh']
        self.topk = config['topk']
        if using_bn:
            self.backbone = VGGBackboneBN(config['backbone']['vgg'], input_channel, device=device)
        else:
            self.backbone = VGGBackbone(config['backbone']['vgg'], input_channel, device=device)

        self.detector_head = DetectorHead(input_channel=128, grid_size=grid_size,using_bn=using_bn)

    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x H x W.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
        """
        if isinstance(x, dict):
            feat_map = self.backbone(x['img'])
        else:
            feat_map = self.backbone(x)
            
        outputs = self.detector_head(feat_map)
        # def forward(self, x):
        # out = None
        # if self.using_bn:
        #     out = self.bnPa(self.relu(self.convPa(x)))
        #     out = self.bnPb(self.convPb(out))  #(B,65,H,W)
        # else:
        #     out = self.relu(self.convPa(x))
        #     out = self.convPb(out)  # (B,65,H,W)

        # prob = self.softmax(out)
        # prob = prob[:, :-1, :, :]  # remove dustbin,[B,64,H,W]
        
        # # Reshape to get full resolution heatmap.
        # prob = pixel_shuffle(prob, self.grid_size)  # [B,1,H*8,W*8]
        # prob = prob.squeeze(dim=1)#[B,H,W]

        # return {'logits':out, 'prob':prob}



        prob = outputs['prob']
        if self.nms is not None:
            prob = [box_nms(p.unsqueeze(dim=0),
                            self.nms,
                            min_prob=self.det_thresh,
                            keep_top_k=self.topk).squeeze(dim=0) for p in prob]
            prob = torch.stack(prob)
            outputs.setdefault('prob_nms',prob)

        pred = prob[prob>=self.det_thresh]

        #print("pred shape in forward :", pred.shape)

        outputs.setdefault('pred', pred)

        return outputs

