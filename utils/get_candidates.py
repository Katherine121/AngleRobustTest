import os
import math
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from load_model import load_fsra, load_rknet, load_lpn

torch.set_printoptions(profile="full")


def get_big_map(path):
    """
    get the big map stored on the UAV locally
    :param path: stored path
    :return: big map paths and center coordinates
    """
    paths = []
    labels = []

    file_path = os.listdir(path)
    file_path.sort()

    for file in file_path:
        full_file_path = os.path.join(path, file)
        paths.append(full_file_path)
        file = file[:-5]
        file = file.split(',')
        labels.append(list(map(eval, [file[0], file[1]])))

    return paths, labels


def get_ideal_route(path):
    """
    get the ideal route
    :param path: file path that recorded all cluster centers
    :return: longitude and latitude coordinates of the ideal route
    """
    route = []
    f = open(path, 'rt')
    for line in f:
        line = line.strip('\n')
        line = line.split(' ')
        route.append(list(map(eval, [line[0], line[1]])))
    route.sort(reverse=True)
    f.close()

    return route


def transform_predicted_label_to_route_idx(path, center):
    """
    transform the predicted label (not sorted) into the position index (sorted) in the ideal route
    :param path: file path that recorded all cluster centers
    :param center: predicted label (not sorted),
                    which should be transformed to the position index (sorted) in the ideal route
    :return: the position index (sorted) in the ideal route
    """
    # the sorted ideal route
    ideal_route = get_ideal_route(path)
    # not sorted labels
    f = open(path, 'rt')
    i = 0
    for line in f:
        if i == center:
            line = line.strip('\n')
            line = line.split(' ')
            # find the position index (sorted) in the ideal route
            route_idx = ideal_route.index(list(map(eval, [line[0], line[1]])))
            break
        i += 1
    f.close()

    return route_idx


def save_candidates(thresh, k_size):
    """
    save candidate image for matching-based methods
    :param thresh: 50 m
    :param k_size: the number of rectangles
    :return:
    """
    if os.path.exists("candidates") is False:
        os.mkdir("candidates")

    paths, labels = get_big_map(path="../bigmap")
    # ideal route coordinates
    centers = get_ideal_route(path="../processOrder/100/cluster_centre.txt")

    lat_diff = thresh * 9e-6
    lon_diff = thresh * 1.043e-5
    number = 0
    # get candidate images of each sorted position
    for ideal_center in centers:
        print(ideal_center)
        candidates = [[ideal_center[0], ideal_center[1]]]
        # thresh * k_size
        for k in range(0, k_size):
            candidates.append([ideal_center[0] - k * lat_diff, ideal_center[1] - k * lon_diff])
            candidates.append([ideal_center[0] - k * lat_diff, ideal_center[1] + k * lon_diff])
            candidates.append([ideal_center[0] + k * lat_diff, ideal_center[1] - k * lon_diff])
            candidates.append([ideal_center[0] + k * lat_diff, ideal_center[1] + k * lon_diff])

        if os.path.exists("candidates/" + str(number)) is False:
            os.mkdir("candidates/" + str(number))

        # screenshot all candidate images from big map
        for candi_i in range(0, len(candidates)):
            if os.path.exists("candidates/" + str(number) + "/" + str(candidates[candi_i][0]) + "," + str(
                    candidates[candi_i][1]) + '.png'):
                continue
            min_dis = math.inf
            idx = -1

            for i in range(0, len(labels)):
                lat_dis = (candidates[candi_i][0] - labels[i][0]) * 111000
                lon_dis = (candidates[candi_i][1] - labels[i][1]) * 111000 * math.cos(labels[i][0] / 180 * math.pi)

                dis = math.sqrt(lat_dis * lat_dis + lon_dis * lon_dis)
                if dis < min_dis:
                    min_dis = dis
                    idx = i

            # find the most match big map and screenshot
            lat_dis = (candidates[candi_i][0] - labels[idx][0]) * 111000
            lon_dis = (candidates[candi_i][1] - labels[idx][1]) * 111000 * math.cos(labels[idx][0] / 180 * math.pi)
            lat_pixel_dis = lat_dis // 0.13986
            lon_pixel_dis = lon_dis // 0.14075
            center = [5005 // 2, 8192 // 2]
            new_lat_pixel = center[0] - lat_pixel_dis
            new_lon_pixel = center[1] + lon_pixel_dis

            pic = Image.open(paths[idx])
            pic = pic.crop((new_lon_pixel - 960 // 2, new_lat_pixel - 540 // 2,
                            new_lon_pixel + 960 // 2, new_lat_pixel + 540 // 2))
            pic = pic.resize((320, 180))
            pic.save("candidates/" + str(number) + "/" + str(candidates[candi_i][0]) + "," + str(
                candidates[candi_i][1]) + '.png')

        number += 1


def save_candidates_tensor(save_dir):
    """
    save the output of the matching-based models when inputting candidate images
    :param save_dir: saved directory of candidate image tenors of one method
    :return:
    """
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)

    val_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # load model
    if "fsra" in save_dir:
        match_model = load_fsra(path="../checkpoint/fsra.pth.tar")
    elif "rknet" in save_dir:
        match_model = load_rknet(path="../checkpoint/rknet.pth.tar")
    else:
        match_model = load_lpn(path="../checkpoint/lpn.pth.tar")
    match_model.eval()

    dirs = os.listdir("./candidates/")
    dirs.sort(key=lambda x: int(x))

    for dir in dirs:
        print(dir)
        # concatenate all candidate images as model input
        candidates_input = None
        candidates_pos = None
        full_dir = os.path.join("./candidates", dir)

        candidates = os.listdir(full_dir)
        # 41
        candidates.sort(key=lambda x: os.path.getctime(os.path.join(full_dir, x)))

        for candi in candidates:
            full_candi = os.path.join(full_dir, candi)

            vals = candi[:-5]
            vals = vals.split(',')
            lat = float(vals[0])
            print(lat)
            lon = float(vals[1])
            pos = np.array((lat, lon))
            pos = np.expand_dims(pos, axis=0)
            print(pos)
            if candidates_pos is None:
                candidates_pos = pos
            else:
                candidates_pos = np.concatenate((candidates_pos, pos), axis=0)

            pic = Image.open(full_candi)
            pic = pic.convert('RGB')
            pic = val_transform(pic).unsqueeze(dim=0)
            # 41
            if candidates_input is None:
                candidates_input = pic
            else:
                candidates_input = torch.cat((candidates_input, pic), dim=0)

        # get features of all candidate images
        candidates_feature, _ = match_model(candidates_input, None)

        if len(candidates_feature.shape) is 3:
            fnorm = torch.norm(candidates_feature, p=2, dim=1, keepdim=True) * np.sqrt(candidates_feature.size(-1))
            candidates_feature = candidates_feature.div(fnorm.expand_as(candidates_feature))
            candidates_feature = candidates_feature.view(candidates_feature.size(0), -1)
        else:
            fnorm = torch.norm(candidates_feature, p=2, dim=1, keepdim=True)
            candidates_feature = candidates_feature.div(fnorm.expand_as(candidates_feature))

        np.savetxt(save_dir + "/" + dir + "_candidates_pos.pt", candidates_pos)
        np.savetxt(save_dir + "/" + dir + "_candidates_feature.pt", candidates_feature.data.cpu().numpy())


if __name__ == '__main__':
    save_candidates(thresh=10, k_size=10)
    save_candidates_tensor(save_dir="fsra")
    save_candidates_tensor(save_dir="rknet")
    save_candidates_tensor(save_dir="lpn")
