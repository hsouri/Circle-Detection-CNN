import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # input images 1 * 200 * 200
        self.L1 = nn.Sequential(nn.Conv2d(1, 32, 5), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2))  # 32 * 98 * 98
        self.L2 = nn.Sequential(nn.Conv2d(32, 64, 3), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2))  # 64 * 48 * 48
        self.L3 = nn.Sequential(nn.Conv2d(64, 128, 3), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2))  # 128 * 23 * 23
        self.L4 = nn.Sequential(nn.Conv2d(128, 128, 3), nn.BatchNorm2d(128), nn.ReLU())  # 128 * 21 * 21
        self.L5 = nn.Sequential(nn.Conv2d(128, 4, 1), nn.BatchNorm2d(4), nn.ReLU())  # 4 * 21 * 21
        self.FC = nn.Sequential(nn.Linear(4 * 21* 21, 256), nn.ReLU(), nn.Linear(256, 16), nn.ReLU())
        self.Last = nn.Linear(16, 3)

    def forward(self, x):
        x = self.L5(self.L4(self.L3(self.L2(self.L1(x)))))
        B, C, H, W = x.shape
        x = x.view(-1, C * H * W)
        x = self.FC(x)
        x = self.Last(x)
        return x
