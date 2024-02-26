import os
import math
import matplotlib.pyplot as plt
import numpy as np
from compute_error import get_one_noise_interp_route, get_ideal_interp_route
from cor import END_LAT, END_LON, NOISE_DB


def draw_MRE(path, a, b):
    """
    draw MRE plot of a method
    :return:
    """
    noise_db = NOISE_DB

    all_interp_lon = []
    all_interp_lat = []
    total_num = []

    # get all kinds of noises
    for i in range(a, b):
        # get all interpolated coordinates of a kind of noise
        interp_lon, interp_lat = get_one_noise_interp_route(path=path,
                                                            noise=noise_db[i][0] + "_" + noise_db[i][1])
        all_interp_lon.extend(interp_lon)
        all_interp_lat.extend(interp_lat)

    # get the ideal route
    ideal_interp_lon, ideal_interp_lat = get_ideal_interp_route(path="../processOrder/100/cluster_centre.txt")
    lat_true = np.array(ideal_interp_lat)

    all_mre = []
    for i in range(0, len(lat_true)):
        all_mre.append(0)
        total_num.append(0)

    for i in range(0, len(all_interp_lat)):
        interp_lat = all_interp_lat[i]
        if len(interp_lat) < 1:
            continue

        lat_true = np.array(ideal_interp_lat)
        interp_lat = np.array(interp_lat)

        lat_true = lat_true * 111000
        interp_lat = interp_lat * 111000

        # limit the comparison range to the minimum length of ideal latitude list
        thresh = min(len(interp_lat), len(lat_true))
        for j in range(0, min(len(interp_lat), len(lat_true))):
            if abs(lat_true[j] - interp_lat[j]) > 200:
                thresh = j
                break

        lat_true = lat_true[:thresh]
        interp_lat = interp_lat[:thresh]

        for k in range(0, thresh):
            # compute route error at each timestamp (the same longitude)
            all_mre[k] += math.fabs(lat_true[k] - interp_lat[k])
            total_num[k] += 1

    lat_true = np.array(ideal_interp_lat)
    lon_true = np.array(ideal_interp_lon)

    # compute mean route error at each timestamp (the same longitude)
    for k in range(0, len(lon_true)):
        if total_num[k] != 0:
            all_mre[k] /= total_num[k]
        else:
            all_mre = all_mre[0:k]
            total_num = total_num[0:k]
            lon_true = lon_true[0:k]
            break
    print(all_mre)
    print(total_num)

    # common comparative baseline
    baseline = []
    for k in range(0, len(lon_true)):
        baseline.append(200)

    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.plot(lon_true, baseline, color='forestgreen', linestyle='--')
    plt.plot(lon_true, all_mre,
             color='skyblue')
    plt.fill_between(lon_true, all_mre, y2=0, color='skyblue')

    plt.legend(frameon=False)
    plt.show()


def draw_visual_test_routes(path_name, a, b):
    """
    draw visual test routes of a method
    :return:
    """
    noise_db = NOISE_DB

    # ideal end point coordinates
    lat_true = END_LAT
    lon_true = END_LON
    # get all kinds of noises
    for i in range(a, b):
        path = os.path.join(path_name, noise_db[i][0] + "_" + noise_db[i][1])
        routes = os.listdir(path)
        routes.sort()
        j = 0

        res = []
        for route in routes:
            if ".txt" in route:
                full_route_path = os.path.join(path, route)
                f = open(full_route_path, "rt")
                for line in f:
                    line = line.strip('\n')
                    line = line.split(' ')
                    flag = int(line[0])
                    if flag in [0, 1, 2]:
                        res.append(flag)
                f.close()
                routes.remove(route)
                break

        routes.sort(key=lambda x: int(x[4:]))
        # get not interpolated routes of a kind of noise
        for route in routes:
            # if ".txt" in route:
            #     continue
            # choose successful routes
            if res[j] == 1 or res[j] == 2:
                # print(j)
                full_route_path = os.path.join(path, route)

                files = os.listdir(full_route_path)
                files.sort(key=lambda x: int(x[:x.index(',')]))
                lat = []
                lon = []
                for file in files:
                    file = file[:-5]
                    file = file.split(',')
                    lat.append(float(file[1]))
                    lon.append(float(file[2]))

                    # compute the end point error on the premise of successful arrival
                    lat_diff = (float(file[1]) - lat_true) * 111000
                    lon_diff = (float(file[2]) - lon_true) * 111000 * math.cos(float(file[1]) / 180 * math.pi)
                    diff = math.sqrt(lat_diff * lat_diff + lon_diff * lon_diff)
                    if diff <= 50:
                        break

                if diff <= 50:
                    plt.plot(lon[int(0 * len(lon)): int(1 * len(lon))],
                             lat[int(0 * len(lon)): int(1 * len(lon))],
                             color='dodgerblue', zorder=200)
                # else:
                #     plt.plot(lon[int(0 * len(lon)): int(1 * len(lon))],
                #              lat[int(0 * len(lon)): int(1 * len(lon))],
                #              color='dodgerblue')
            j += 1

    # draw the ideal route
    lat = []
    lon = []
    route = []
    f = open("../processOrder/100/cluster_centre.txt", 'rt')
    for line in f:
        line = line.strip('\n')
        line = line.split(' ')
        route.append(list(map(eval, [line[0], line[1]])))
    route.sort(reverse=True)
    f.close()

    for i in range(0, len(route)):
        lat.append(route[i][0])
        lon.append(route[i][1])

    plt.plot(lon[int(0 * len(lon)): int(1 * len(lon))],
             lat[int(0 * len(lon)): int(1 * len(lon))],
             color='black', zorder=201)
    plt.legend(frameon=False)
    plt.show()


def draw_noise_abl():
    """
    draw ablation curves as noise magnitude increases
    :return:
    """
    plt.figure(figsize=(8, 4))
    plt.rcParams['savefig.dpi'] = 400
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 20
    plt.rcParams['font.sans-serif'] = ['Times New Roman']

    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # x = [0, 5, 10]
    # y_our = [1, 0.8, 0.6]
    # y_fsra = [0.07, 0., 0.]
    # y_rknet = [0.03, 0, 0.]
    # y_lpn = [0.04, 0.05, 0]

    # x = [0, 1, 1.5]
    # y_our = [1, 0.5, 0.15]
    # y_fsra = [0.07, 0.1, 0.]
    # y_rknet = [0.03, 0.05, 0.]
    # y_lpn = [0.04, 0.05, 0.15]

    # x = [0, 40, 60]
    # y_our = [1, 0.95, 0.85]
    # y_fsra = [0.07, 0.05, 0.05]
    # y_rknet = [0.03, 0., 0.]
    # y_lpn = [0.04, 0.05, 0.]

    x = [0, 0.2, 0.4]
    y_our = [0.96, 0.95, 0.85]
    y_fsra = [0.07, 0.05, 0.]
    y_rknet = [0.03, 0., 0.]
    y_lpn = [0.04, 0.05, 0.]
    plt.plot(x, y_our, color='red', marker='o', label="Ours")
    plt.plot(x, y_fsra, color='blue', marker='x', label="FSRA")
    plt.plot(x, y_rknet, color='green', marker='*', label="RK-Net")
    plt.plot(x, y_lpn, color='orange', marker='D', label="LPN")
    plt.ylabel("SR@50")

    plt.gcf().subplots_adjust(left=0.2)
    plt.legend(frameon=False)
    plt.show()


def draw_three_loss():
    """
    draw three loss weights: sigma1, sigma2, sigma3 curves as epoch increases
    :return:
    """
    plt.figure(figsize=(8, 4))
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.sans-serif'] = ['Times New Roman']

    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    fp1 = open('lossweight.txt', 'r')
    loss1 = [1]
    loss2 = [1]
    loss3 = [1]
    i = 0
    for loss in fp1:
        if i == 116:
            break
        loss = loss.strip('\n')
        loss = loss.split(' ')
        loss1.append(loss[0])
        loss2.append(loss[1])
        loss3.append(loss[2])
        i += 1
    fp1.close()

    loss1 = np.array(loss1, dtype=float)
    loss2 = np.array(loss2, dtype=float)
    loss3 = np.array(loss3, dtype=float)

    fig, ax = plt.subplots()
    x = np.linspace(0, i, i + 1)
    y3 = loss3
    ax.plot(x, y3, label=r'$\sigma_1$')
    y1 = loss1
    ax.plot(x, y1, label=r'$\sigma_2$')
    y2 = loss2
    ax.plot(x, y2, label=r'$\sigma_3$')

    plt.legend(loc="upper right")
    plt.gcf().subplots_adjust(left=0.15)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.show()


if __name__ == '__main__':
    path = "../test_our_model"
    a = 0
    b = 1
    # a = 1
    # b = len(NOISE_DB)
    # draw_MRE(path, a, b)
    draw_visual_test_routes(path, a, b)
    # draw_noise_abl()
    # draw_three_loss()
