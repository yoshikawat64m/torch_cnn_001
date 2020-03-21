import torch.nn as nn
import torch.nn.functional as F

from util.functions import flatten


class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, 11, stride=4)
        self.max_pool1 = nn.MaxPool2d((3, 3), stride=2)

        self.conv2 = nn.Conv2d(96, 256, 5, stride=1, padding=2)
        self.max_pool2 = nn.MaxPool2d((3, 3), stride=2)

        self.conv3 = nn.Conv2d(256, 384, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 256, 3, stride=1, padding=1)
        self.max_pool3 = nn.MaxPool2d((3, 3), stride=2)

        self.fc1 = nn.Linear(9216, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)

    def forward(self, x):
        # (N, 3, 227, 227)
        x = F.relu(self.conv1(x))   # => (N, 96, 55, 55)
        x = self.max_pool1(x)       # => (N, 96, 27, 27)

        x = F.relu(self.conv2(x))   # => (N, 256, 27, 27)
        x = self.max_pool2(x)       # => (N, 256, 13, 13)

        x = F.relu(self.conv3(x))   # => (N, 384, 13, 13)
        x = F.relu(self.conv4(x))   # => (N, 256, 13, 13)
        x = self.max_pool3(x)       # => (N, 256, 6, 6)

        x = flatten(x)         # => (N, 9216)

        # (N, 9216)
        x = F.relu(self.fc1(x))     # => (N, 4096)
        x = F.relu(self.fc2(x))     # => (N, 4096)
        x = F.softmax(self.fc3(x))  # => (N, 1000)

        # (N, 1000)
        return x
