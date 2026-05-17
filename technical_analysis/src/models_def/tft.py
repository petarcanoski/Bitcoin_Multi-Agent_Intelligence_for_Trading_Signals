import torch
import torch.nn as nn
import torch.nn.functional as F


class GRN(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout=0.1):
        super().__init__()
        self.fc1  = nn.Linear(d_in, d_hidden)
        self.fc2  = nn.Linear(d_hidden, d_out * 2)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = F.elu(self.fc1(x))
        h = self.drop(self.fc2(h))
        h1, h2 = h.chunk(2, dim=-1)
        h = h1 * torch.sigmoid(h2)
        return self.norm(h + self.skip(x))


class TFTClassifier(nn.Module):
    def __init__(self, input_features, d_model=128, n_heads=4,
                 n_lstm_layers=2, dropout=0.1, num_classes=2,
                 use_regression_head=False):
        super().__init__()
        self.use_reg = use_regression_head

        self.vsn_gate = nn.Sequential(
            nn.Linear(input_features, input_features),
            nn.Softmax(dim=-1),
        )
        self.feat_proj = nn.Linear(input_features, d_model)

        self.lstm = nn.LSTM(
            d_model, d_model, n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )
        self.post_lstm = GRN(d_model, d_model, d_model, dropout)

        self.attn      = nn.MultiheadAttention(d_model, n_heads,
                                                dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        self.post_attn = GRN(d_model, d_model, d_model, dropout)

        self.clf = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
        self.reg = (nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(),
                                   nn.Linear(32, 1), nn.Softplus())
                    if use_regression_head else None)

    def forward(self, x):
        gates  = self.vsn_gate(x)
        h      = self.feat_proj(gates * x)

        h, _        = self.lstm(h)
        h           = self.post_lstm(h)

        h_a, _      = self.attn(h, h, h)
        h           = self.attn_norm(h + h_a)
        h           = self.post_attn(h)

        h = h.mean(dim=1)
        return self.clf(h), (self.reg(h) if self.reg else None)
