from collections import namedtuple
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from models.inception.components.util import (
    BasicConv2d,
    InceptionA,
    InceptionB,
    InceptionC,
    InceptionD,
    InceptionE,
    InceptionAux
)


__all__ = ['InceptionV3', 'inception_v3']


model_urls = {
    # Inception v3 ported from TensorFlow
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}

_InceptionOuputs = namedtuple('InceptionOuputs', ['logits', 'aux_logits'])


def inception_v3(pretrained=False, **kwargs):
    """Inception v3 model architecture
    """
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True

        if 'aux_logits' in kwargs:
            original_aux_logits = kwargs['aux_logits']
            kwargs['aux_logits'] = True
        else:
            original_aux_logits = True

        model = InceptionV3(**kwargs)
        model.load_state_dict(model_zoo.load_url(model_urls['inception_v3_google']))

        if not original_aux_logits:
            model.aux_logits = False
            del model.AuxLogits

        return model

    return InceptionV3(**kwargs)


class InceptionV3(nn.Module):

    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False):
        super(InceptionV3, self).__init__()

        self.aux_logits = aux_logits
        self.transform_input = transform_input

        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)

        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)

        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)

        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)

        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)

                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())

                with torch.no_grad():
                    m.weight.copy_(values)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)

        # (N, 3, 299, 299)
        x = self.Conv2d_1a_3x3(x)  # -> (N, 32, 149, 149)
        x = self.Conv2d_2a_3x3(x)  # -> (N, 32, 147, 147)
        x = self.Conv2d_2b_3x3(x)  # -> (N, 64, 147, 147)
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        # (N, 64, 73, 73)
        x = self.Conv2d_3b_1x1(x)  # -> (N, 80, 73, 73)
        x = self.Conv2d_4a_3x3(x)  # -> (N, 192, 71, 71)
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        # (N, 192, 35, 35)
        x = self.Mixed_5b(x)  # -> (N, 256, 35, 35)
        x = self.Mixed_5c(x)  # -> (N, 288, 35, 35)
        x = self.Mixed_5d(x)  # -> (N, 288, 35, 35)
        x = self.Mixed_6a(x)  # -> (N, 768, 17, 17)
        x = self.Mixed_6b(x)  # -> (N, 768, 17, 17)
        x = self.Mixed_6c(x)  # -> (N, 768, 17, 17)
        x = self.Mixed_6d(x)  # -> (N, 768, 17, 17)
        x = self.Mixed_6e(x)

        # (N, 768, 17, 17)
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)

        # (N, 768, 17, 17)
        x = self.Mixed_7a(x)  # -> (N, 1280, 8, 8)
        x = self.Mixed_7b(x)  # -> (N, 2048, 8, 8)
        x = self.Mixed_7c(x)

        # (N, 2048, 8, 8)
        x = F.adaptive_avg_pool2d(x, (1, 1))

        # (N, 2048, 1, 1)
        x = F.dropout(x, training=self.training)

        # (N, 2048, 1, 1)
        x = x.view(x.size(0), -1)

        # (N, 2048)
        x = self.fc(x)

        # (N, 1000) (num_classes)
        if self.training and self.aux_logits:
            return _InceptionOuputs(x, aux)

        return x
