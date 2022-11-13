"""
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
"""
import logging
from typing import Callable, List, Union

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor

logger = logging.getLogger(__file__)

allow_stochastic_depth = True
try:
    from torchvision.ops import StochasticDepth
except ImportError:
    logger.warn(
        "Cannot import `torchvision.ops.StochasticDepth`, disable this feature."
    )
    allow_stochastic_depth = False

__all__ = [
    "ResNet",
    "resnet20",
    "resnet32",
    "resnet44",
    "resnet56",
    "resnet110",
    "resnet1202",
]


def _weights_init(m: Union[nn.Linear, nn.Conv2d]) -> None:
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd: Callable) -> None:
        super().__init__()
        self.lambd = lambd

    def forward(self, x: Tensor) -> Tensor:
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        option: str = "A",
        stochastic_depth_p: float = 0.0,
        stochastic_depth_mode: str = "row",
        bn_layer: nn.modules.batchnorm._BatchNorm = nn.BatchNorm2d,
        act_fn: Callable = F.relu,
    ) -> None:

        super().__init__()
        self.act_fn = act_fn
        if not allow_stochastic_depth or stochastic_depth_p == 0.0:
            self.stochastic_depth = None
        else:
            self.stochastic_depth = StochasticDepth(
                stochastic_depth_p, stochastic_depth_mode
            )

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = bn_layer(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = bn_layer(planes)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != planes:
            if option == "A":
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes // 4, planes // 4),
                        "constant",
                        0,
                    )
                )
            elif option == "B":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    bn_layer(self.expansion * planes),
                )

    def forward(self, x: Tensor) -> Tensor:
        out = self.act_fn(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.stochastic_depth is not None:
            self.stochastic_depth(out)
        out += self.shortcut(x)
        out = self.act_fn(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: int,
        num_blocks: List[int],
        num_classes: int = 10,
        stochastic_depth_p: float = 0.0,
        stochastic_depth_mode: str = "row",
        bn_layer: nn.modules.batchnorm._BatchNorm = nn.BatchNorm2d,
        act_fn: Callable = F.relu,
    ) -> None:
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = bn_layer(16)
        self.act_fn = act_fn

        self.layer1 = self._make_layer(
            block,
            16,
            num_blocks[0],
            1,
            stochastic_depth_p,
            stochastic_depth_mode,
            bn_layer,
            act_fn,
        )
        self.layer2 = self._make_layer(
            block,
            32,
            num_blocks[1],
            2,
            stochastic_depth_p,
            stochastic_depth_mode,
            bn_layer,
            act_fn,
        )
        self.layer3 = self._make_layer(
            block,
            64,
            num_blocks[2],
            2,
            stochastic_depth_p,
            stochastic_depth_mode,
            bn_layer,
            act_fn,
        )
        self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)

    def _make_layer(
        self,
        block: BasicBlock,
        planes: int,
        num_blocks: int,
        stride: int,
        stochastic_depth_p: float = 0.0,
        stochastic_depth_mode: str = "row",
        bn_layer: nn.modules.batchnorm._BatchNorm = nn.BatchNorm2d,
        act_fn: Callable = F.relu,
    ) -> nn.Sequential:

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    stride,
                    "A",
                    stochastic_depth_p,
                    stochastic_depth_mode,
                    bn_layer,
                    act_fn,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = self.act_fn(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20() -> ResNet:
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32() -> ResNet:
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44() -> ResNet:
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56() -> ResNet:
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110() -> ResNet:
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202() -> ResNet:
    return ResNet(BasicBlock, [200, 200, 200])


def test(net: ResNet) -> None:
    import numpy as np

    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print(
        "Total layers",
        len(
            list(
                filter(
                    lambda p: p.requires_grad and len(p.data.size()) > 1,
                    net.parameters(),
                )
            )
        ),
    )


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith("resnet"):
            print(net_name)
            test(globals()[net_name]())
            print()
