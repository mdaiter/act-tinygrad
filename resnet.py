
from config import *
from utils import *
from layers import *
from networks import *

from tinygrad import Tensor, dtypes, nn
from tinygrad.helpers import fetch, get_child

# allow monkeypatching in layer implementations
Conv2d = nn.Conv2d
Linear = nn.Linear

class FrozenBatchNorm2d:
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        # Register buffers instead of parameters
        self.weight = Tensor.ones(num_features, requires_grad=False)
        self.bias = Tensor.zeros(num_features, requires_grad=False)
        self.running_mean = Tensor.zeros(num_features, requires_grad=False)
        self.running_var = Tensor.ones(num_features, requires_grad=False)
    def __call__(self, x:Tensor) -> Tensor:
        # Reshape for 2D input
        scale = (self.weight / (self.running_var + self.eps).sqrt()).reshape(1, -1, 1, 1)
        bias = (self.bias - self.running_mean * scale.flatten()).reshape(1, -1, 1, 1)
        return x * scale + bias

class Block:
    def __init__(self, in_dims, dims, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_dims, dims, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = FrozenBatchNorm2d(dims)
        self.conv2 = nn.Conv2d(
            dims, dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = FrozenBatchNorm2d(dims)
        self.downsample = []
        if stride != 1:
            self.downsample = [
                nn.Conv2d(in_dims, dims, kernel_size=1, stride=stride, bias=False),
                FrozenBatchNorm2d(dims)
            ]

    def __call__(self, x:Tensor) -> Tensor:
        base_operations = [
            self.conv1,
            self.bn1,
            Tensor.relu,
            self.conv2,
            self.bn2
        ]
        out = x.sequential(base_operations)
    
        if self.downsample != []:
            return (x.sequential(base_operations) + x.sequential(self.downsample)).relu()
        else:
            return x.sequential(base_operations).relu()

class ResNet:
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = FrozenBatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, 512, num_blocks[3], stride=2)
        #self.fc = nn.Linear(512, num_classes, requires_grad=False) # if we decide to use this someday, remove the grad
    def _make_layer(self, block, in_dims, dims, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(in_dims, dims, stride))
            in_dims = dims
        return layers
    def __call__(self, x:Tensor):
        x = self.bn1(self.conv1(x)).relu().max_pool2d()
        x = x.sequential(self.layer1)
        x = x.sequential(self.layer2 + self.layer3 + self.layer4)
        """
        Commented out for now, because we're just using the output from layer4
        """
        #x = x.mean([2, 3])
        #x = self.fc(x)
        return x

class ResNetInstances:
    def resnet18_IMAGENET1K_V1_Generator():
        resnet18_IMAGENET1K = ResNet(Block, [2, 2, 2, 2], num_classes=1000)
        state_dict = nn.state.safe_load("resnet18-f37072fd.safetensors")
        nn.state.load_state_dict(resnet18_IMAGENET1K, state_dict)
        return resnet18_IMAGENET1K

    # Static instantiation of resnet18, so we don't have to create it multiple times
    resnet18_IMAGENET1K_V1 = resnet18_IMAGENET1K_V1_Generator()
