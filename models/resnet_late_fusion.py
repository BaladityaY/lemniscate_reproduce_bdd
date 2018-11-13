import torch
from torch.autograd import Variable
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from lib.normalize import Normalize
import numpy as np


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, low_dim=128):
        self.inplanes = 64
        super(ResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer_concat(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, low_dim)
        self.l2norm = Normalize(2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    # Custom make layer for steering concat
    def _make_layer_concat(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes+18, planes * block.expansion, #steering concat
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes+18, planes, stride, downsample)) #steering concat
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def _get_flow_fields(self, w_y_trips, t_z=1):
        all_flow_fields = None

        for w_y_trip in w_y_trips:
            flow_fields = None

            for w_y in w_y_trip:
                t_x = 0
                t_y = 0
                w_x = 0
                w_z = 0

                x_len = y_len = 28; norm_factor = ((x_len)/2.)**2 + 1.
                xs = np.arange(x_len) - x_len/2.
                ys = np.arange(y_len) - y_len/2.

                XX, YY = np.meshgrid(xs, -ys)

                u_field = np.zeros((x_len, y_len))
                v_field = np.zeros((x_len, y_len))

                for x_ind in range(x_len):
                    for y_ind in range(y_len):
                        x = XX[x_ind, y_ind]; y = YY[x_ind, y_ind]

                        t_vec = np.array([t_x, t_y, t_z])
                        t_x_comp = np.array([-1, 0, y])
                        t_y_comp = np.array([0, -1, x])

                        w_vec = np.array([w_x, w_y, w_z])
                        w_x_comp = np.array([x*y, -1*(1 + x**2), y])
                        w_y_comp = np.array([1 + y**2, -1*x*y, -1*x])

                        u_field[x_ind, y_ind] = np.dot(t_x_comp, t_vec) + np.dot(w_x_comp, w_vec)
                        v_field[x_ind, y_ind] = np.dot(t_y_comp, t_vec) + np.dot(w_y_comp, w_vec)

                u_field = u_field/norm_factor
                v_field = v_field/norm_factor

                mag_field = u_field**2 + v_field**2

                flow_field = np.zeros((x_len, y_len, 3))
                flow_field[:,:,0] = u_field
                flow_field[:,:,1] = v_field
                flow_field[:,:,2] = mag_field

                flow_field = flow_field.reshape((1,x_len,y_len,3))

                if flow_fields is None:
                    flow_fields = flow_field
                else:
                    flow_fields = np.concatenate((flow_fields, flow_field), axis=3)

            if all_flow_fields is None:
                all_flow_fields = flow_fields
            else:
                all_flow_fields = np.concatenate((all_flow_fields, flow_fields), axis=0)

        return all_flow_fields

    def forward(self, imgs, action_probabilities):
        #imgs = imgs[:,6:,:,:]
        print("imgs shape: {}".format(imgs.shape))
        #print("action_probabilities shape: {}".format(action_probabilities.shape))
        
        imgs = self.conv1(imgs)
        
        #print('post conv1')

        imgs = self.bn1(imgs)
        imgs = self.relu(imgs)
        imgs = self.maxpool(imgs)

        imgs = self.layer1(imgs)
        imgs = self.layer2(imgs)
        
        #print('post layer 2')
        
        # Design decision. Chosen position for late fusion of action_probabilities command
        # Steering commands for each timestep are put in separate channels and then
        # copied to occupy the entire channel
        #print('pre action_probabilities size: {}'.format(action_probabilities.size()))
        #print('pre action_probabilities vals: {}'.format(action_probabilities))
        
        
        ### CREATE STEERING HOTMATRICES
#         og_steering = action_probabilities.clone()
#         action_probabilities = action_probabilities + Variable(torch.Tensor([1.0]).float().cuda()).expand(action_probabilities.size())
#         action_probabilities = torch.round(action_probabilities * Variable(torch.Tensor([6.0/2.0]).float().cuda()).expand(action_probabilities.size())).long()
#         action_probabilities[action_probabilities < 0] = 0
#         action_probabilities[action_probabilities > 6] = 6
#         #print('post action_probabilities size: {}'.format(action_probabilities.size()))
#         #print('post action_probabilities vals: {}'.format(action_probabilities))
#         
#         steering_hotvecs = None
#         for i in range(3):
#             steering_hotvec = torch.zeros(action_probabilities.size()[0], 7, 1)
#             #print('og action_probabilities i: {} \nsteering i: {}'.format(og_steering[:,i], action_probabilities[:,i]))
#             steering_hotvec = steering_hotvec.scatter(1, action_probabilities[:,i].cpu().view(action_probabilities.size()[0],1,1), 1)
#             
#             steering_hotvec = steering_hotvec.permute(0, 2, 1)
#             
#             if steering_hotvecs is None:
#                 steering_hotvecs = steering_hotvec
#             else:
#                 steering_hotvecs = torch.cat((steering_hotvecs, steering_hotvec), dim=1)
#                 
#         steering_hotvecs = steering_hotvecs.view(steering_hotvecs.size()[0],
#                                                  steering_hotvecs.size()[1],
#                                                  steering_hotvecs.size()[2], 1)
#         
#         steering_hotvecs = steering_hotvecs.permute(2, 3, 0, 1)
#                 
#         #print('steering_hotvecs shape: {}'.format(steering_hotvecs.shape))
#         #print('steering_hotvecs vals: {}'.format(steering_hotvecs))
#         
#         steering_hotvecs = steering_hotvecs.expand(-1, 28, -1, -1)
#         
#         steering_hotvecs = torch.cat((steering_hotvecs, steering_hotvecs, 
#                                       steering_hotvecs, steering_hotvecs), dim=0)
#         
#         steering_hotvecs = steering_hotvecs.permute(2, 3, 0, 1).cuda()
#         

        #print('steering_hotvecs shape post expansion: {}'.format(steering_hotvecs.shape))
        #print('steering_hotvecs vals post expansion: {}'.format(steering_hotvecs))
        
        #for i in range(3):
        #    print('val: {}'.format(steering_hotvecs[0,i,:14,:14]))
        
        # ORIGINAL WAY OF ADDING STEER
        action_probabilities = action_probabilities.view(action_probabilities.size()[0],-1).expand(28,28,-1,-1).permute(2,3,0,1)
        
        #print action_probabilities
        #print('action_probabilities size: {}'.format(action_probabilities.size()))
        #print('imgs size: {}'.format(imgs.size()))

        imgs = torch.cat((imgs, action_probabilities), dim=1)
        
        imgs = self.layer3(imgs)
        imgs = self.layer4(imgs)
        
        imgs = self.avgpool(imgs)
        imgs = imgs.view(imgs.size(0), -1)
        imgs = self.fc(imgs)
        imgs = self.l2norm(imgs)

        return imgs


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
