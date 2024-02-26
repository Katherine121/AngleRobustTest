import argparse
import math
import os.path
import random
import time
import numpy as np
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from cor import END_LAT, END_LON, START_LAT, START_LON, NOISE_DB, START_SHIFT, HEIGHT_SHIFT
from load_dataset import *
from load_model import *
from utils.data_augment import ImageAugment
from utils.get_candidates import *

torch.set_printoptions(precision=8)


parser = argparse.ArgumentParser(description='PyTorch Testing')

parser.add_argument('--num_classes1', default=100, type=int,
                    metavar='N', help='the number of position labels')
parser.add_argument('--num_classes2', default=2, type=int,
                    metavar='N', help='the number of angle labels (latitude and longitude)')
parser.add_argument('--len', default=6, type=int,
                    metavar='LEN', help='the number of model input sequence length')
parser.add_argument('--height', default=540, type=float,
                    metavar='LEN', help='h')
parser.add_argument('--width', default=960, type=float,
                    metavar='LEN', help='w')
parser.add_argument('--dis', default=60, type=float,
                    metavar='LEN', help='dis of once fly')
parser.add_argument('--thresh', default=25, type=float,
                    metavar='LEN', help='the maximum distance from the end point')
parser.add_argument('--test_num', default=10, type=int,
                    metavar='LEN', help='test number of each path of each method')

parser.add_argument('--model-resume', default='checkpoint/vit.pth.tar', type=str,
                    metavar='PATH', help='path to latest checkpoint (default: none)')

parser.add_argument('--save_dir', default='test_vit/', type=str,
                    metavar='PATH', help='path to captured images')
parser.add_argument('--start_path', default='',
                    type=list,
                    metavar='PATH',
                    help='the frame path of the start point')
parser.add_argument('--dest_path', default="",
                    type=str,
                    metavar='PATH',
                    help='the frame path of the end point')
parser.add_argument('--last_pos', default=[],
                    type=list,
                    metavar='ANGLE',
                    help='last position: lat and lon')
parser.add_argument('--dest_pos', default=[],
                    type=list,
                    metavar='ANGLE',
                    help='the end point position: lat and lon')


def main():
    """
    realistic testing process control of classification-based method
    :return:
    """
    args = parser.parse_args()
    save_dir = args.save_dir
    # get big map stored in the UAV locally
    paths, labels = get_big_map(path="./bigmap")
    # get the ideal route
    route = get_ideal_route(path="./processOrder/100/cluster_centre.txt")

    # all kinds of noises
    noise_db = NOISE_DB

    # load model
    if "vit" in args.save_dir:
        class_model = load_vit(args)
    elif "resnet" in args.save_dir:
        class_model = load_resnet(args)
    else:
        class_model = load_mobilenet(args)
    class_model.eval()
    # reduce CPU usage, use it after the model is loaded onto the GPU
    torch.set_num_threads(1)
    cudnn.benchmark = True

    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)

    # start testing for all kinds of noises
    for i in range(0, len(noise_db)):

        args.save_dir = save_dir + noise_db[i][0] + "_" + noise_db[i][1]
        print(args.save_dir)
        if os.path.exists(args.save_dir) is False:
            os.mkdir(args.save_dir)

        # choose a kind of noise
        image_augment = ImageAugment(style_idx=noise_db[i][0], shift_idx=noise_db[i][1])

        if noise_db[i][0] == "ori" and noise_db[i][1] == "ori":
            args.test_num = 100
        else:
            args.test_num = 20
        print(args.test_num)

        top1 = 0
        top5 = 0
        total_test_num = 0

        # start testing routes for a kind of noise
        for j in range(0, args.test_num):

            args.save_dir = save_dir + noise_db[i][0] + "_" + noise_db[i][1] + "/test" + str(j)
            if os.path.exists(args.save_dir) is False:
                os.mkdir(args.save_dir)

            # initialize the start point and end point for each route
            init_input(paths, labels, image_augment, args)

            # whether reach the end point or not, the frame path of reaching the end point, inference time
            arrive, flag, avg_infer_time = test_class_model(class_model, route, paths, labels, image_augment, args)

            if arrive == 1:
                top1 += 1
            elif arrive == 2:
                top5 += 1
            total_test_num += 1

            with open(save_dir + noise_db[i][0] + "_" + noise_db[i][1] + "/result.txt", "a") as file1:
                file1.write(str(arrive) + " " + flag + " " + str(avg_infer_time) + "\n")
            file1.close()

        with open(save_dir + noise_db[i][0] + "_" + noise_db[i][1] + "/result.txt", "a") as file1:
            file1.write(str(top1) + " " + str(top5) + " " +
                        str(total_test_num) + "\n")
        file1.close()


def init_input(paths, labels, image_augment, args):
    """
    initialize the start point and end point for each route
    :param paths: big map paths
    :param labels: big map labels
    :param image_augment: noise
    :param args:
    :return:
    """
    args.last_pos = [START_LAT, START_LON]
    args.dest_pos = [END_LAT, END_LON]

    # the realistic start point: random 25m
    args.last_pos[0] += random.uniform(START_SHIFT * -9e-6, START_SHIFT * 9e-6)
    args.last_pos[1] += random.uniform(START_SHIFT * -1.043e-5, START_SHIFT * 1.043e-5)

    # randomly add shift noise to the start point
    shift = image_augment.forward_shift()
    args.last_pos[0] += shift[0]
    args.last_pos[1] += shift[1]
    # screenshot the start point image
    args.start_path = screenshot(paths, labels, 0, args.last_pos[0], args.last_pos[1], image_augment, args)

    args.dest_path = str(END_LAT) + "," + str(END_LON) + ".png"


def test_class_model(class_model, route, paths, labels, image_augment, args):
    """
    realistic testing process of a route
    :param class_model: classify model
    :param route: the sorted ideal route
    :param paths: big map paths
    :param labels: big map labels
    :param image_augment: image augment
    :param args:
    :return: whether reach the end point or not, the frame path of reaching the end point, inference time
    """
    flag = ""
    total_infer_time = 0

    with torch.no_grad():
        for i in range(1, 100):
            # load model input
            images = load_class_dataset(args)

            # b,3,224,224
            images = images.to(dtype=torch.float32)

            start_time = time.time()
            output = class_model(images)
            _, center = output.max(1)

            # transform the predicted label (not sorted) into the position index (sorted) in the ideal route
            center = transform_predicted_label_to_route_idx(path="./processOrder/100/cluster_centre.txt",
                                                            center=center)
            # if predicted route index is the end point
            if center == 99:
                lat_diff = (args.last_pos[0] - route[-1][0]) * 111000
                lon_diff = (args.last_pos[1] - route[-1][1]) * 111000 * math.cos(route[-1][0] / 180 * math.pi)
                diff = math.sqrt(lat_diff * lat_diff + lon_diff * lon_diff)
                # calculate the difference between the actual position and the end point
                if diff <= args.thresh:
                    print("You reach the end point successfully in 25m!")
                    return 1, args.start_path, total_infer_time / i
                if diff <= 2 * args.thresh:
                    print("You reach the end point successfully in 50m!")
                    flag = args.start_path
                else:
                    return 0, "None", total_infer_time / i

            # calculate the moving distance at the next time
            lat_delta = route[i][0] - route[center][0]
            lon_delta = route[i][1] - route[center][1]

            # calculate the inference time
            end_time = time.time()
            run_time = (end_time - start_time) * 1000
            total_infer_time += run_time

            # calculate the new position at the next time
            new_lat = args.last_pos[0] + lat_delta
            new_lon = args.last_pos[1] + lon_delta

            # add shift noise to the origin position
            shift = image_augment.forward_shift()
            # print(shift)
            new_lat += shift[0]
            new_lon += shift[1]

            # screenshot new frame
            path = screenshot(paths, labels, i, new_lat, new_lon, image_augment, args)
            if path == -1:
                # successful arrival within 50 m
                if flag != "":
                    return 2, flag, total_infer_time / i
                # fail
                else:
                    print("You fly away!")
                    return 0, "None", total_infer_time / i
            else:
                # append new frame and new angle
                args.start_path = path
                args.last_pos = [new_lat, new_lon]

                lat_diff = (new_lat - route[-1][0]) * 111000
                lon_diff = (new_lon - route[-1][1]) * 111000 * math.cos(route[-1][0] / 180 * math.pi)
                diff = math.sqrt(lat_diff * lat_diff + lon_diff * lon_diff)
                # successful arrival within 25 m
                if diff <= args.thresh:
                    print("You reach the end point successfully in 25m!")
                    return 1, path, total_infer_time / i
                # successful arrival within 50 m
                if diff <= 2 * args.thresh:
                    print("You reach the end point successfully in 50m!")
                    flag = path

    # successful arrival within 50 m
    if flag != "":
        return 2, flag, total_infer_time / i
    # fail
    else:
        print("You fly away!")
        return 0, "None", total_infer_time / i


def screenshot(paths, labels, number, new_lat, new_lon, img_aug, args):
    """
    screenshot an image based on ideal or adjusted coordinates
    :param paths: big map paths
    :param labels: big map labels
    :param number: the index of the image in the current route
    :param new_lat: the latitude of the screenshot
    :param new_lon: the longitude of the screenshot
    :param image_aug: image augment
    :param args:
    """
    # new frame path
    path = args.save_dir + '/' + str(number) + "," + str(new_lat) + "," + str(new_lon) + '.png'
    if os.path.exists(path) is False:
        min_dis = math.inf
        idx = -1

        for i in range(0, len(labels)):
            lat_dis = (new_lat - labels[i][0]) * 111000
            lon_dis = (new_lon - labels[i][1]) * 111000 * math.cos(labels[i][0] / 180 * math.pi)

            dis = math.sqrt(lat_dis * lat_dis + lon_dis * lon_dis)
            if dis < min_dis:
                min_dis = dis
                idx = i

        # latitude: 700 m
        # longitude: 1153 m
        # 5005, 8192
        # 0.13986 lat m / pixel
        # 0.14075 lpn m / pixel
        # 1.82 alt m / lat m
        # 1.82 alt m / lon m
        # 0.2548 alt m / pixel
        # 0.2548 alt m / pixel
        # find the most match big map and screenshot
        lat_dis = (new_lat - labels[idx][0]) * 111000
        lon_dis = (new_lon - labels[idx][1]) * 111000 * math.cos(labels[idx][0] / 180 * math.pi)
        lat_pixel_dis = lat_dis // 0.13986
        lon_pixel_dis = lon_dis // 0.14075
        center = [5005 // 2, 8192 // 2]
        new_lat_pixel = center[0] - lat_pixel_dis
        new_lon_pixel = center[1] + lon_pixel_dis

        # altitude changing -40~40m, pixel changing
        if img_aug.hei:
            height = args.height + random.uniform(a=-HEIGHT_SHIFT / 0.2548, b=HEIGHT_SHIFT / 0.2548)
            width = args.width + random.uniform(a=-HEIGHT_SHIFT / 0.2548, b=HEIGHT_SHIFT / 0.2548)
        else:
            height = args.height
            width = args.width

        # If the center of the new image is out of bounds
        if new_lon_pixel - width // 2 > 8192:
            return -1
        if new_lat_pixel - height // 2 > 5005:
            return -1
        if new_lon_pixel + width // 2 < 0:
            return -1
        if new_lat_pixel + height // 2 < 0:
            return -1

        pic = Image.open(paths[idx])
        pic = pic.crop((new_lon_pixel - width // 2, new_lat_pixel - height // 2,
                        new_lon_pixel + width // 2, new_lat_pixel + height // 2))
        pic = pic.resize((320, 180))

        # add style noise to the new image
        pic = np.array(pic)
        pic = img_aug(pic)
        pic = Image.fromarray(pic)
        pic.save(path)

    # new screenshot image
    return path


if __name__ == '__main__':
    main()
