import torch
import torch.nn as nn


class RNNClassifier(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        # input shape: (B, L, D)

        self.stem = nn.Linear(embed_dim, 1024)

        # self.rnn = nn.GRU(input_size=embed_dim, hidden_size=128, num_layers=3, batch_first=True)
        self.rnn = nn.LSTM(input_size=1024, hidden_size=512, num_layers=1, batch_first=True, bidirectional=True)
        # (B, L=512, C=512)

        self.fc = nn.Sequential(
            nn.Linear(1024, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),   # (B, L=512, C=1)
            nn.ReLU(inplace=True),
            nn.Flatten(),   # (B, 512)
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, L, D)
        x = self.stem(x)
        x, _ = self.rnn(x)
        # x: (B, L, 128)
        return self.fc(x)


if __name__ == "__main__":
    model = RNNClassifier(768)
    x = torch.randn(32, 512, 768)
    y = model(x)
    print(y.shape)
    torch.save(model.state_dict(), "model.pth")