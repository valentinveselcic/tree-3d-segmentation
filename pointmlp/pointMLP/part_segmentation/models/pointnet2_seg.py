import torch
import torch.nn as nn
from models.pointnet2_part_seg_msg import get_model

class PointNet2SegWrapper(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # normal_channel=True → expects 6-dim input (xyz+rgb)
        self.model = get_model(num_classes=num_classes, normal_channel=True)
        self.num_classes = num_classes

    def forward(self, points, feat):
        """
        points : (B, 3, N)  xyz          — unused directly, feat contains xyz
        feat   : (B, 6, N)  xyz + rgb
        Returns logits (B, C, N)
        """
        B = feat.shape[0]
        N = feat.shape[2]
        cls_label = torch.zeros(B, 16, device=feat.device)   # dummy, no category
        
        out, _ = self.model(feat, cls_label)  # out: (B, N, C), log-softmax already applied
        return out.permute(0, 2, 1)           # → (B, C, N)  