import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, label_size=10):
        """Shallow convolutional neural network
        label_size (int): number of classes in dataset.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        self.drop1 = nn.Dropout2d(p=0.3)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.drop2 = nn.Dropout2d(p=0.3)
        self.pool2 = nn.MaxPool2d(2)
        self.head_linear = nn.Linear(1568, label_size)

    def forward(self, input_batch):
        out = self.conv1(input_batch)
        out = self.drop1(out)
        out = self.pool1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.drop2(out)
        out = self.pool2(out)
        out = F.relu(out)
        out_flat = out.view(out.shape[0], -1)
        logits = self.head_linear(out_flat)
        probabilities = F.softmax(logits, dim=1)
        return logits, probabilities

    def _init_weights(self):
        """override pytorch initialization with kaiming normalization
        """
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv3d,
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d
            }:
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu',
                )
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)
