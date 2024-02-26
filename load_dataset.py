import torch
from PIL import Image
from torchvision import transforms


def load_our_dataset(args):
    """
    generate model inputs
    :param args:
    :return: frame sequence and angle sequence
    """

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    # slice to avoid modifying the original list
    start_path = args.start_path[:]
    next_angles = args.next_angles[:]

    next_imgs = None

    # original images and angles
    for i in range(0, len(start_path)):
        img = start_path[i]
        img = Image.open(img)
        img = img.convert('RGB')
        img = val_transform(img).unsqueeze(dim=0)
        if next_imgs is None:
            next_imgs = img
        else:
            next_imgs = torch.cat((next_imgs, img), dim=0)
    next_angles.append([0, 0])

    # append the end point frame as part of model input
    dest_img = Image.open(args.dest_path)
    dest_img = dest_img.convert('RGB')
    dest_img = val_transform(dest_img).unsqueeze(dim=0)
    next_imgs = torch.cat((next_imgs, dest_img), dim=0)

    dest_angle = [0, 0]
    next_angles.append(dest_angle)
    next_angles = torch.tensor(next_angles, dtype=torch.float)

    # if there are not enough input frames
    for i in range(0, args.len - 1 - len(start_path)):
        next_imgs = torch.cat((next_imgs, torch.zeros((1, 3, 224, 224))), dim=0)
        next_angles = torch.cat((next_angles, torch.zeros((1, 2))), dim=0)

    # add a batch dimension
    return next_imgs.unsqueeze(dim=0), next_angles.unsqueeze(dim=0)


def load_class_dataset(args):
    """
    generate model inputs
    :param args:
    :return: frame
    """

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    img = args.start_path
    img = Image.open(img)
    img = img.convert('RGB')
    img = val_transform(img)

    # add a batch dimension
    return img.unsqueeze(dim=0)


def load_match_dataset(val_transform, args):
    """
    generate model inputs
    :param val_transform: torchvision.transforms
    :param args:
    :return: frame
    """
    img = args.start_path
    img = Image.open(img)
    img = img.convert('RGB')
    img = val_transform(img)

    # add a batch dimension
    return img.unsqueeze(dim=0)
