import os
import torch
from torch import nn
import torchvision.models as torchvision_models
from torchvision.models import MobileNet_V3_Small_Weights

from checkpoint.bs_models import vit, resnet18, mobilenet_v3
from checkpoint.lpn import lpn_two_view_net
from checkpoint.model import ARTransformer
from checkpoint.fsra import fsra_two_view_net
from checkpoint.rknet import rknet_two_view_net


def load_resnet(args):
    """
    load resnet18
    :param args:
    :return:
    """
    model = resnet18(num_classes1=args.num_classes1)
    args.model_resume = "checkpoint/resnet.pth.tar"
    # load from resume, start training from a certain epoch
    if args.model_resume:
        if os.path.isfile(args.model_resume):
            print("=> loading checkpoint '{}'".format(args.model_resume))
            checkpoint = torch.load(args.model_resume, map_location='cpu')

            state_dict = checkpoint['state_dict']
            new_state_dict = state_dict
            for key in list(state_dict.keys()):
                new_state_dict[key[len("module."):]] = state_dict[key]
                del new_state_dict[key]
            model.load_state_dict(new_state_dict, strict=False)

            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            best_acc2 = checkpoint['best_acc2']
            best_acc3 = checkpoint['best_acc3']
            print("best_acc1: " + str(best_acc1))
            print("best_acc2: " + str(best_acc2))
            print("best_acc3: " + str(best_acc3))

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.model_resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.model_resume))

    return model


def load_mobilenet(args):
    """
    load mobilenetv3 small
    :param args:
    :return:
    """
    model = mobilenet_v3(num_classes1=args.num_classes1)
    args.model_resume = "checkpoint/mobilenet.pth.tar"
    # load from resume, start training from a certain epoch
    if args.model_resume:
        if os.path.isfile(args.model_resume):
            print("=> loading checkpoint '{}'".format(args.model_resume))
            checkpoint = torch.load(args.model_resume, map_location='cpu')

            state_dict = checkpoint['state_dict']
            new_state_dict = state_dict
            for key in list(state_dict.keys()):
                new_state_dict[key[len("module."):]] = state_dict[key]
                del new_state_dict[key]
            model.load_state_dict(new_state_dict, strict=False)

            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            best_acc2 = checkpoint['best_acc2']
            best_acc3 = checkpoint['best_acc3']
            print("best_acc1: " + str(best_acc1))
            print("best_acc2: " + str(best_acc2))
            print("best_acc3: " + str(best_acc3))

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.model_resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.model_resume))

    return model


def load_vit(args):
    """
    load vit
    :param args:
    :return:
    """
    model = vit(num_classes1=args.num_classes1)
    args.model_resume = "checkpoint/vit.pth.tar"
    # load from resume, start training from a certain epoch
    if args.model_resume:
        if os.path.isfile(args.model_resume):
            print("=> loading checkpoint '{}'".format(args.model_resume))
            checkpoint = torch.load(args.model_resume, map_location='cpu')

            state_dict = checkpoint['state_dict']
            new_state_dict = state_dict
            for key in list(state_dict.keys()):
                new_state_dict[key[len("module."):]] = state_dict[key]
                del new_state_dict[key]
            model.load_state_dict(new_state_dict, strict=False)

            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            best_acc2 = checkpoint['best_acc2']
            best_acc3 = checkpoint['best_acc3']
            print("best_acc1: " + str(best_acc1))
            print("best_acc2: " + str(best_acc2))
            print("best_acc3: " + str(best_acc3))

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.model_resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.model_resume))

    return model


def load_lpn(path=None):
    """
    load lpn
    :return:
    """
    model = lpn_two_view_net(class_num=100, droprate=0.75, stride=1, share_weight=True, block=2)
    if path:
        model.load_state_dict(torch.load(path), strict=False)
    else:
        model.load_state_dict(torch.load("checkpoint/lpn.pth.tar"), strict=False)
    changeclassify(model)
    # 4096
    print(model)
    return model


def load_rknet(path=None):
    """
    load rknet
    :return:
    """
    model = rknet_two_view_net(class_num=100, droprate=0.65, stride=1, share_weight=True)
    if path:
        model.load_state_dict(torch.load(path), strict=False)
    else:
        model.load_state_dict(torch.load("checkpoint/rknet.pth.tar"), strict=False)
    changeclassify(model)
    # 2048
    print(model)
    return model


def load_fsra(path=None):
    """
    load fsra
    :return:
    """
    model = fsra_two_view_net(class_num=100, block=3)
    if path:
        model.load_state_dict(torch.load(path))
    else:
        model.load_state_dict(torch.load("checkpoint/fsra.pth.tar"))
    changeclassify(model)
    # 3072
    print(model)
    return model


def changeclassify(model):
    for name, module in model.named_children():
        if "classifier" in name:
            setattr(model, name, nn.Identity())
        else:
            changeclassify(module)


def load_our_model(args):
    """
    load our angle robustness model
    :param args:
    :return:
    """
    # backbone = torchvision_models.shufflenet_v2_x0_5(weights=(ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1))
    backbone = torchvision_models.mobilenet_v3_small(MobileNet_V3_Small_Weights.IMAGENET1K_V1)

    model = ARTransformer(
        backbone=backbone,
        extractor_dim=576,
        num_classes1=args.num_classes1,
        num_classes2=args.num_classes2,
        len=args.len,
        dim=512,
        depth=4,
        heads=8,
        dim_head=64,
        mlp_dim=1024,
        dropout=0.1,
        emb_dropout=0.1
    )

    # load from resume, start training from a certain epoch
    if args.model_resume:
        if os.path.isfile(args.model_resume):
            print("=> loading checkpoint '{}'".format(args.model_resume))
            checkpoint = torch.load(args.model_resume, map_location='cpu')

            state_dict = checkpoint['state_dict']
            new_state_dict = state_dict
            for key in list(state_dict.keys()):
                new_state_dict[key[len("module."):]] = state_dict[key]
                del new_state_dict[key]
            model.load_state_dict(new_state_dict, strict=False)

            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            best_acc2 = checkpoint['best_acc2']
            best_acc3 = checkpoint['best_acc3']
            print("best_acc1: " + str(best_acc1))
            print("best_acc2: " + str(best_acc2))
            print("best_acc3: " + str(best_acc3))

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.model_resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.model_resume))

    return model
