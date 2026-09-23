import torch
from torch import nn


# Layers	Output resolution
# Conv7×7 s2 3→32, MaxPool 3×3 s2 p1	S/4
# Conv5×5 32→64	S/4
# Conv3×3 s2 64→128	S/8
# Conv1×1 128→256	S/8
# Conv3×3 s2 256→256	S/16
# Conv1×1 256→512	S/16
# head: GlobalAvgPool, Linear 512→256, ReLU, Linear 256→100

class MyModel(nn.Module):
    def __init__(self, out_dim: int = 100):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2, padding=7//2, bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=5 // 2, stride=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=3 // 2, stride=2, bias=False)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, padding=1 // 2, stride=1, bias=False)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=3 // 2, stride=2, bias=False)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, padding=1 // 2, stride=1, bias=False)

        self.pool2 = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc1 = nn.Linear(in_features=512, out_features=256)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=256, out_features=out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool2(x)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
