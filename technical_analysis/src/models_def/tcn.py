import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, dropout=0.1):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.c1  = nn.utils.weight_norm(nn.Conv1d(in_ch,  out_ch, kernel, dilation=dilation))
        self.c2  = nn.utils.weight_norm(nn.Conv1d(out_ch, out_ch, kernel, dilation=dilation))
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = F.relu(self.drop(self.c1(F.pad(x, (self.pad, 0)))))
        h = F.relu(self.drop(self.c2(F.pad(h, (self.pad, 0)))))
        return F.relu(h + self.skip(x))


class TCNModel(nn.Module):
    def __init__(self, input_features, hidden=128, kernel_size=3,
                 n_blocks=6, dropout=0.1, num_classes=2,
                 use_regression_head=False):
        super().__init__()
        self.use_reg = use_regression_head
        dilations = [2 ** i for i in range(n_blocks)]

        blocks = [nn.Conv1d(input_features, hidden, 1)]
        ch = hidden
        for d in dilations:
            blocks.append(_ResBlock(ch, hidden, kernel_size, d, dropout))
            ch = hidden
        self.net = nn.Sequential(*blocks)

        self.clf = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
        self.reg = (nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(),
                                   nn.Linear(32, 1), nn.Softplus())
                    if use_regression_head else None)

    def forward(self, x):
        h = self.net(x.permute(0, 2, 1))
        h = h.mean(dim=-1)
        return self.clf(h), (self.reg(h) if self.reg else None)
