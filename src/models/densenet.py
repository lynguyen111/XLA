import torch
import torch.nn as nn

class DenseLayer(nn.Module):
    """Một layer trong DenseBlock: BN-ReLU-Conv1x1-BN-ReLU-Conv3x3 (bottleneck)."""

    def __init__(self, in_channels, growth_rate, bn_size=4, dropout=0.0):
        super().__init__()
        mid_channels = bn_size * growth_rate
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, growth_rate, kernel_size=3, padding=1, bias=False),
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x):
        out = self.block(x)
        if self.dropout:
            out = self.dropout(out)
        # Dense connection: concat input với output
        return torch.cat([x, out], dim=1)


class DenseBlock(nn.Module):
    """Tập hợp num_layers DenseLayer liên tiếp."""

    def __init__(self, num_layers, in_channels, growth_rate, bn_size=4, dropout=0.0):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(
                DenseLayer(in_channels + i * growth_rate, growth_rate, bn_size, dropout)
            )
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class TransitionBlock(nn.Module):
    """Giảm chiều channel và spatial: BN-ReLU-Conv1x1-AvgPool2x2."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class DenseNet(nn.Module):
    def __init__(
        self,
        block_config,  # số DenseLayer mỗi block, vd [6,12,24,16]
        growth_rate=32,
        num_init_features=64,
        bn_size=4,
        dropout=0.0,
        num_classes=13,
        theta=0.5,  # compression ratio ở TransitionBlock
    ):
        super().__init__()

        # Stem: Conv7x7 + BN + ReLU + MaxPool
        self.stem = nn.Sequential(
            nn.Conv2d(
                3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False
            ),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        layers = []
        in_channels = num_init_features

        for i, num_layers in enumerate(block_config):
            layers.append(
                DenseBlock(num_layers, in_channels, growth_rate, bn_size, dropout)
            )
            in_channels = in_channels + num_layers * growth_rate

            # Thêm Transition sau mỗi DenseBlock trừ block cuối
            if i < len(block_config) - 1:
                out_channels = int(in_channels * theta)
                layers.append(TransitionBlock(in_channels, out_channels))
                in_channels = out_channels

        # BN cuối trước classifier
        layers.append(nn.BatchNorm2d(in_channels))
        layers.append(nn.ReLU(inplace=True))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_channels, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def get_densenet121(num_classes=13, dropout=0.2):
    """DenseNet-121: growth_rate=32, block_config=[6,12,24,16]."""
    return DenseNet(
        block_config=[6, 12, 24, 16],
        growth_rate=32,
        num_init_features=64,
        dropout=dropout,
        num_classes=num_classes,
    )


def get_densenet169(num_classes=13, dropout=0.2):
    """DenseNet-169: growth_rate=32, block_config=[6,12,32,32]."""
    return DenseNet(
        block_config=[6, 12, 32, 32],
        growth_rate=32,
        num_init_features=64,
        dropout=dropout,
        num_classes=num_classes,
    )


def get_densenet201(num_classes=13, dropout=0.2):
    """DenseNet-201: growth_rate=32, block_config=[6,12,48,32]."""
    return DenseNet(
        block_config=[6, 12, 48, 32],
        growth_rate=32,
        num_init_features=64,
        dropout=dropout,
        num_classes=num_classes,
    )


