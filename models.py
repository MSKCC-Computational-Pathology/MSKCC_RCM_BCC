import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import conv3x3, BasicBlock

layers = [3, 4, 6, 3]

def _make_layer(block, inplanes, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            conv1x1(inplanes, planes * block.expansion, stride),
            nn.BatchNorm2d(planes * block.expansion),
        )
    
    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(inplanes, planes))
    
    return nn.Sequential(*layers)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def get_model(zdim, nclasses):
    m = models.resnet34(True)
    layer4 = _make_layer(BasicBlock, 256, zdim, layers[3], stride=2)
    m.layer4 = layer4
    m.fc = nn.Linear(zdim, nclasses)
    m = m.cuda()
    return m

class deepsup(nn.Module):
    def __init__(self):
        super(deepsup, self).__init__()
        model = models.resnet34(True)
        conv1 = model._modules['conv1'].weight.detach().clone().mean(dim=1, keepdim=True)
        model._modules['conv1'] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model._modules['conv1'].weight.data = conv1
        self.features1 = nn.Sequential(*list(model.children())[0:-2])
        self.features2 = nn.Sequential(
            _make_layer(BasicBlock, 512, 256, 3, stride=2),
            _make_layer(BasicBlock, 256, 128, 3, stride=2)
        )
        self.avgpool1 = nn.AvgPool2d(32)
        self.avgpool2 = nn.AvgPool2d(8)
        self.fc1 = nn.Linear(512, 2)
        self.fc2 = nn.Linear(128, 2)
    
    def forward(self, x):
        x = self.features1(x)
        o1 = self.avgpool1(x)
        o1 = self.fc1(o1.view(o1.size(0), -1))
        o2 = self.features2(x)
        o2 = self.avgpool2(o2)
        o2 = self.fc2(o2.view(o2.size(0), -1))
        return o1, o2

def get_embedder(zdim, nclasses, path):
    model = get_model(zdim, nclasses)
    ch = torch.load(path)
    model.load_state_dict(ch['state_dict'])
    # Extract embedder
    return embedder(model)

