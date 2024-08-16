# Creator Cui Liz
# Time 29/07/2024 21:32

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ResBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(in_c, out_c * 2, 3, 1, 1),
            nn.BatchNorm1d(out_c * 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(out_c * 2, out_c, 3, 1, 1),
        )

        self.shortcut = nn.Conv1d(in_c, out_c, 1, 1, 0) if in_c != out_c else nn.Identity()

    def forward(self, x):
        return self.layers(x) + self.shortcut(x)


class ResClassifier(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        # input : (B, 512, 768)
        # (B, 768, 512)
        self.layers = nn.Sequential(
            nn.Conv1d(embed_dim, 512, 3, 1, 1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(2),
            ResBlock(512, 512),
            ResBlock(512, 256),
            ResBlock(256, 128),
            nn.Conv1d(128, 16, 3, 1, 1),

            nn.MaxPool1d(2),

            nn.Flatten(),   # (B, 512)
            nn.Linear(2048, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x.transpose(1, 2))


if __name__ == "__main__":
    model = ResClassifier(768)
    x = torch.randn(2, 512, 768)
    y = model(x)
    print(y.shape)
    torch.save(model.state_dict(), "Res.pth")