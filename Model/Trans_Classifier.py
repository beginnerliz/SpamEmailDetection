# Creator Cui Liz
# Time 13/08/2024 02:46

import torch
import torch.nn as nn
from einops import rearrange
from math import sqrt

class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__(
            nn.Conv1d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(inplace=True),
        )


class Transpose(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.transpose(1, 2)


class BasicBlock(nn.Module):
    def __init__(self, in_c: int, nheads: int):
        super().__init__()

        self.enabled = False

        self.H = nheads

        self.qkv_proj = nn.Linear(in_c, in_c * 3)

        self.scale = 1 / sqrt(in_c)

        self.linear = nn.Linear(in_c, in_c)

        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        # B, L, C = x.shape
        if not self.enabled:
            return x

        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)

        q = rearrange(q, 'b l (h c) -> (b h) l c', h=self.H)
        kt = rearrange(k, 'b l (h c) -> (b h) c l', h=self.H)
        v = rearrange(v, 'b l (h c) -> (b h) l c', h=self.H)

        attn = torch.softmax(torch.bmm(q, kt) * self.scale, dim=-1)     # (B*H, L, L)

        heads = rearrange(torch.bmm(attn, v), '(b h) l c -> b l (h c)', h=self.H)

        return self.linear(heads) + x



class TransformerClassifier(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.layers = nn.Sequential(
            Transpose(),
            ConvBlock(embed_dim, 1024),
            ConvBlock(1024, 512),
            nn.MaxPool1d(2),
            BasicBlock(256, 4),
            BasicBlock(256, 4),
            ConvBlock(512, 256),
            BasicBlock(256, 4),
            BasicBlock(256, 4),
            ConvBlock(256, 128),
            ConvBlock(128, 32),
            nn.MaxPool1d(2),
            Transpose(),
        )

        self.rnn = nn.LSTM(input_size=32, hidden_size=32, num_layers=1, batch_first=True)
        nn.init.zeros_(self.rnn.weight_ih_l0)
        nn.init.zeros_(self.rnn.weight_hh_l0)
        nn.init.zeros_(self.rnn.bias_ih_l0)
        nn.init.zeros_(self.rnn.bias_hh_l0)

        self.fc = nn.Sequential(    # (B, L, 512)
            nn.Linear(32, 32),
            nn.LeakyReLU(inplace=True),
            nn.Linear(32, 16),
            nn.LeakyReLU(inplace=True),
            nn.Flatten(),
        )

        self.out = nn.Sequential(
            nn.Linear(2048, 1024),  # (B, 128)
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.all_enabled = False

    def enableAll(self):
        for i in range(len(self.layers)):
            if isinstance(self.layers[i], BasicBlock):
                self.layers[i].enabled = True
        self.all_enabled = True

    def onlyCNN(self):
        for i in range(len(self.layers)):
            if isinstance(self.layers[i], BasicBlock):
                self.layers[i].enabled = True
        self.all_enabled = False

    def forward(self, x):
        # input (B, L, C)
        x = self.layers(x)          # (B, C, L) -> (B, L, C) for transformer encoder
        # trans_out = self.trans(x)
        # x = torch.cat([cnn_out, trans_out], dim=2)
        if self.all_enabled:
            x = self.rnn(x)[0] + x
        x = self.fc(x)
        return self.out(x)


if __name__ == "__main__":
    model = TransformerClassifier(768)
    x = torch.randn(2, 512, 768)
    y = model(x)
    print(y.shape)

    # loss_func = nn.BCELoss()
    # target = torch.randint(0, 2, (2, 1)).float()

    # loss = loss_func(y, target)
    # loss.backward()

    torch.save(model.state_dict(), "model.pth")