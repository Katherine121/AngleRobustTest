import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.nn import functional as F


######################################################################
class USAM(nn.Module):
    def __init__(self, kernel_size=3, padding=1, polish=True):
        super(USAM, self).__init__()

        kernel = torch.ones((kernel_size, kernel_size))
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

        kernel2 = torch.ones((1, 1)) * (kernel_size * kernel_size)
        kernel2 = kernel2.unsqueeze(0).unsqueeze(0)
        self.weight2 = nn.Parameter(data=kernel2, requires_grad=False)

        self.polish = polish
        self.pad = padding
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(1)

    def __call__(self, x):
        fmap = x.sum(1, keepdim=True)
        x1 = F.conv2d(fmap, self.weight, padding=self.pad)
        x2 = F.conv2d(fmap, self.weight2, padding=0)

        att = x2 - x1
        att = self.bn(att)
        att = self.relu(att)

        if self.polish:
            att[:, :, :, 0] = 0
            att[:, :, :, -1] = 0
            att[:, :, 0, :] = 0
            att[:, :, -1, :] = 0

        output = x + att * x

        return output


class GeM(nn.Module):
    # GeM zhedong zheng
    def __init__(self, dim=2048, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(dim) * p, requires_grad=True)  # initial p
        self.eps = eps
        self.dim = dim

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        x = torch.transpose(x, 1, -1)
        x = x.clamp(min=eps).pow(p)
        x = torch.transpose(x, 1, -1)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        x = x.view(x.size(0), x.size(1))
        x = x.pow(1. / p)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ',' + 'dim=' + str(self.dim) + ')'


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


def fix_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True,
                 return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x, f
        else:
            x = self.classifier(x)
            return x


class ft_net_VGG16(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(ft_net_VGG16, self).__init__()
        model_ft = models.vgg16_bn(pretrained=True)
        # avg pooling to global pooling
        # if stride == 1:
        #    model_ft.layer4[0].downsample[0].stride = (1,1)
        #    model_ft.layer4[0].conv2.stride = (1,1)

        self.pool = pool
        if pool == 'avg+max':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))
            # self.classifier = ClassBlock(4096, class_num, droprate)
        elif pool == 'avg':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
            # self.classifier = ClassBlock(2048, class_num, droprate)
        elif pool == 'max':
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))
        elif pool == 'gem':
            model_ft.gem2 = GeM(dim=512)

        self.model = model_ft

        if init_model != None:
            self.model = init_model.model
            self.pool = init_model.pool
            # self.classifier.add_block = init_model.classifier.add_block

    def forward(self, x):
        x = self.model.features(x)
        if self.pool == 'avg+max':
            x1 = self.model.avgpool2(x)
            x2 = self.model.maxpool2(x)
            x = torch.cat((x1, x2), dim=1)
        elif self.pool == 'avg':
            x = self.model.avgpool2(x)
        elif self.pool == 'max':
            x = self.model.maxpool2(x)
        elif self.pool == 'gem':
            x = self.model.gem2(x)

        x = x.view(x.size(0), x.size(1))

        # x = self.classifier(x)
        return x


# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1, 1)
            model_ft.layer4[0].conv2.stride = (1, 1)

        self.pool = pool
        if pool == 'avg+max':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))
            # self.classifier = ClassBlock(4096, class_num, droprate)
        elif pool == 'avg':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
            # self.classifier = ClassBlock(2048, class_num, droprate)
        elif pool == 'max':
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))
        elif pool == 'gem':
            model_ft.gem2 = GeM(dim=2048)

        self.model = model_ft

        if init_model != None:
            self.model = init_model.model
            self.pool = init_model.pool
            # self.classifier.add_block = init_model.classifier.add_block

        self.usam_1 = USAM()
        self.usam_2 = USAM()

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.usam_1(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.usam_2(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        if self.pool == 'avg+max':
            x1 = self.model.avgpool2(x)
            x2 = self.model.maxpool2(x)
            x = torch.cat((x1, x2), dim=1)
        elif self.pool == 'avg':
            x = self.model.avgpool2(x)
        elif self.pool == 'max':
            x = self.model.maxpool2(x)
        elif self.pool == 'gem':
            x = self.model.gem2(x)

        x = x.view(x.size(0), x.size(1))
        # x = self.classifier(x)
        return x


class rknet_two_view_net(nn.Module):
    def __init__(self, class_num, droprate, stride = 2, pool = 'avg', share_weight = False, VGG16=False):
        super(rknet_two_view_net, self).__init__()
        if VGG16:
            self.model_1 =  ft_net_VGG16(class_num, stride=stride, pool = pool)
        else:
            self.model_1 =  ft_net(class_num, stride=stride, pool = pool)
        if share_weight:
            self.model_2 = self.model_1
        else:
            if VGG16:
                self.model_2 =  ft_net_VGG16(class_num, stride = stride, pool = pool)
            else:
                self.model_2 =  ft_net(class_num, stride = stride, pool = pool)

    def forward(self, x1, x2):
        if x1 is None:
            y1 = None
        else:
            y1 = self.model_1(x1)

        if x2 is None:
            y2 = None
        else:
            y2 = self.model_2(x2)
        return y1, y2
