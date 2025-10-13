import torch
import torch.nn as nn

class SlotHead(nn.Module):
    def __init__(self, d_in=64, d_hid=128, num_classes=47, dropout=0.0):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(d_in, d_hid),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cls = nn.Linear(d_hid, num_classes)

    def logits(self, f):
        B,S,D = f.shape
        h = self.feat(f.view(B*S, D))
        return self.cls(h).view(B, S, -1)

    def penultimate(self, f):
        B,S,D = f.shape
        return self.feat(f.view(B*S, D)).view(B,S,-1)
