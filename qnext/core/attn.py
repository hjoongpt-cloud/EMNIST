import torch
import torch.nn as nn

class SlotMHSA(nn.Module):
    def __init__(self, d, heads=2, dropout=0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, slot_emb):
        x = self.norm(slot_emb)
        out, _ = self.mha(query=x, key=x, value=x, need_weights=False)
        return slot_emb + self.gate * out

class LocalSelfAttn(nn.Module):
    def __init__(self, d, kernel_size=3):
        super().__init__()
        self.pad = kernel_size//2
        self.depthwise = nn.Conv1d(d, d, kernel_size=kernel_size, padding=self.pad, groups=d, bias=False)
        self.norm = nn.LayerNorm(d)
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, slot_emb):
        x = self.norm(slot_emb)
        x1 = x.transpose(1,2)
        y = self.depthwise(x1).transpose(1,2)
        return slot_emb + self.gate * y
