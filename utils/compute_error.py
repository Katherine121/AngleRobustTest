import math
import os
import numpy as np
from sklearn.metrics import mean_absolute_error

from cor import END_LAT, END_LON, NOISE_DB


def get_interp_route(lon, lat):
    """
    interpolate a route every 10 m in longitude orientation
    :param lon: origin longitude coordinates
    :param lat: origin latitude coordinates
    :return: longitude and latitude coordinates after being interpolated
    """
    interp_lon = np.linspace(min(lon), max(lon), int((max(lon) - min(lon)) / 1.0434393e-4))
    interp_lat = np.interp(interp_lon, lon, lat)

    return interp_lon, interp_lat


def get_one_noise_interp_route(path="test_our_model", noise="ori_ori"):
    """
    read all routes of a kind of noisy directory and interpolate all routes
    :param path: method name
    :param noise: noise name
    :return: all routes after being interpolated of a kind of noisy directory
    """
    path = os.path.join(path, noise)
    routes = os.listdir(path)
    routes.sort()

    all_interp_lon = []
    all_interp_lat = []

    for route in routes:
        if ".txt" in route:
            continue
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
        # interpolate a route
        if len(lon) > 1:
            interp_lon, interp_lat = get_interp_route(lon, lat)
        else:
            interp_lon = lon
            interp_lat = lat
        all_interp_lon.append(interp_lon)
        all_interp_lat.append(interp_lat)

    return all_interp_lon, all_interp_lat


def get_ideal_interp_route(path="../processOrder/100/cluster_centre.txt"):
    """
    get the ideal interpolated route
    :param path: file path that recorded all cluster centers
    :return: interpolated longitude and latitude coordinates of the ideal route
    """
    ideal_xy = []
    f = open(path, 'rt')
    for line in f:
        line = line.strip('\n')
        line = line.split(' ')
        ideal_xy.append(list(map(eval, [line[0], line[1]])))
    ideal_xy.sort(reverse=True)
    f.close()

    lat = []
    lon = []
    for i in range(0, len(ideal_xy)):
        lat.append(ideal_xy[i][0])
        lon.append(ideal_xy[i][1])
    ideal_interp_lon, ideal_interp_lat = get_interp_route(lon, lat)
    return ideal_interp_lon, ideal_interp_lat


def get_one_noise_res(path, noise):
    """
    get the result txt of one method and one kind of noise
    :param path: method name
    :param noise: noise name
    :return: [whether reach the end point or not, the frame path of reaching the end point, inference time],
             [the number of reaching the place less than 25 m from the end point,
             the number of reaching the place less than 50 m from the end point,
             the number of routes].
    """
    path = os.path.join(path, noise)
    routes = os.listdir(path)
    routes.sort()

    one_noise_res = []
    for route in routes:
        if ".txt" in route:
            full_res_path = os.path.join(path, route)
            f = open(full_res_path, 'rt')
            for line in f:
                line = line.strip('\n')
                line = line.split(' ')
                one_noise_res.append(line)
            f.close()

    return one_noise_res[0: -1], [one_noise_res[-1]]


def get_success_rate(path, a, b):
    """
    compute SR@25 and SR@50 of a method
    :return: the successful rate of reaching the place less than 25 m from the end point,
             the successful rate of reaching the place less than 50 m from the end point
    """
    noise_db = NOISE_DB

    all_noise_res = []
    top1 = 0
    top5 = 0
    total_num = 0

    # get all kinds of noises
    for i in range(a, b):
        # get the result txt of a kind of noise
        _, one_noise_res = get_one_noise_res(path=path,
                                          noise=noise_db[i][0] + "_" + noise_db[i][1])
        all_noise_res.extend(one_noise_res)

    # read the result txts in all noisy directories
    for i in range(0, len(all_noise_res)):
        top1 += int(all_noise_res[i][0])
        top5 += int(all_noise_res[i][1])
        total_num += int(all_noise_res[i][2])

    print(float(top1 / total_num))
    print(float((top1 + top5) / total_num))


def get_MEPE(path, a, b):
    """
    compute Mean End Point Error(MEPE) of a method
    :return: Mean End Point Error(MEPE) on the premise of successful arrival
    """
    noise_db = NOISE_DB

    all_noise_res = []
    top1_diff = 0
    top5_diff = 0
    total_num = 0

    # get all kinds of noises
    for i in range(a, b):
        # get the result txt of a kind of noise
        one_noise_res, _ = get_one_noise_res(path=path,
                                             noise=noise_db[i][0] + "_" + noise_db[i][1])
        all_noise_res.extend(one_noise_res)

    lat_true = END_LAT
    lon_true = END_LON

    # process all result txts
    for i in range(0, len(all_noise_res)):

        # read whether reach the end point or not
        flag = all_noise_res[i][0]
        # read the frame path of reaching the end point
        path = all_noise_res[i][1]

        # if success within 25 m
        if flag == '1':
            path = path.split('/')
            path = path[-1]
            path = path[:-5]
            path = path.split(',')
            # read the frame coordinates of reaching the end point
            lat_preds = float(path[1])
            lon_preds = float(path[2])

            # compute the end point error on the premise of successful arrival
            lon_diff = (lon_preds - lon_true) * 111000 * math.cos(lat_true / 180 * math.pi)
            lat_diff = (lat_preds - lat_true) * 111000
            diff = math.sqrt(lon_diff * lon_diff + lat_diff * lat_diff)
            top1_diff += diff
            total_num += 1

        # if success within 50 m
        if flag == '2':
            path = path.split('/')
            path = path[-1]
            path = path[:-5]
            path = path.split(',')
            # read the frame coordinates of reaching the end point
            lat_preds = float(path[1])
            lon_preds = float(path[2])

            # compute the end point error on the premise of successful arrival
            lon_diff = (lon_preds - lon_true) * 111000 * math.cos(lat_true / 180 * math.pi)
            lat_diff = (lat_preds - lat_true) * 111000
            diff = math.sqrt(lon_diff * lon_diff + lat_diff * lat_diff)
            top5_diff += diff
            total_num += 1

    # compute mean end point error on the premise of successful arrival
    if total_num != 0:
        print(float(top1_diff / total_num))
        print(float((top1_diff + top5_diff) / total_num))


def get_MRE(path, a, b):
    """
    compute Mean Route Error(MRE) of a method
    :return: Mean Route Error(MRE) on the premise of not deviating from the prescribed route
    We set the maximum deviation range to 200 m
    """
    noise_db = NOISE_DB

    all_interp_lon = []
    all_interp_lat = []
    all_maae = 0
    total_num = 0

    # get all kinds of noises
    for i in range(a, b):
        # get all interpolated routes of a kind of noise
        interp_lon, interp_lat = get_one_noise_interp_route(path=path,
                                                            noise=noise_db[i][0] + "_" + noise_db[i][1])
        all_interp_lon.extend(interp_lon)
        all_interp_lat.extend(interp_lat)

    # get the ideal route
    ideal_interp_lon, ideal_interp_lat = get_ideal_interp_route(path="../processOrder/100/cluster_centre.txt")

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
            # find the index of the frame that deviates from the ideal route more than 200 m
            if abs(lat_true[j] - interp_lat[j]) > 200:
                thresh = j
                break

        lat_true = lat_true[:thresh]
        interp_lat = interp_lat[:thresh]

        # compute RE on the premise of not deviating from the prescribed route
        all_maae += mean_absolute_error(lat_true, interp_lat)
        total_num += 1

    print(all_maae // total_num)
    # compute MRE on the premise of not deviating from the prescribed route
    return all_maae // total_num


def get_MRD(path, a, b):
    """
    compute Mean Route Distance(MRD) of a method
    :return: Mean Route Distance(MRD) on the premise of not deviating from the prescribed route
    We set the maximum deviation range to 200 m
    """
    noise_db = NOISE_DB

    all_interp_lon = []
    all_interp_lat = []
    all_diff = 0
    total_num = 0

    # get all kinds of noises
    for i in range(a, b):
        # get all interpolated routes of a kind of noise
        interp_lon, interp_lat = get_one_noise_interp_route(path=path,
                                                            noise=noise_db[i][0] + "_" + noise_db[i][1])
        all_interp_lon.extend(interp_lon)
        all_interp_lat.extend(interp_lat)

    # get the ideal route
    ideal_interp_lon, ideal_interp_lat = get_ideal_interp_route(path="../processOrder/100/cluster_centre.txt")

    for i in range(0, len(all_interp_lat)):
        interp_lon = all_interp_lon[i]
        interp_lat = all_interp_lat[i]
        if len(interp_lat) < 1:
            continue

        lon_true = np.array(ideal_interp_lon)
        lat_true = np.array(ideal_interp_lat)
        interp_lat = np.array(interp_lat)

        # There are three situations, one is successful arrival within 25 m,
        # one is deviating before successful arrival,
        # and the other is successful arrival within 50 m but continue fly until deviating
        # This is suitable for situations that deviating before successful arrival
        diff = None
        for j in range(0, min(len(interp_lat), len(lat_true))):
            # if deivating
            if abs(lat_true[j] - interp_lat[j]) * 111000 > 200:
                lon_diff = (interp_lon[j - 1] - interp_lon[0]) * 111000 * math.cos(interp_lat[0] / 180 * math.pi)
                lat_diff = (lat_true[j - 1] - lat_true[0]) * 111000
                # compute RD on the premise of not deviating from the prescribed route
                diff = math.sqrt(lon_diff * lon_diff + lat_diff * lat_diff)
                all_diff += diff
                total_num += 1
                break

        # This is suitable for situations that is successful arrival within 25 m
        # or
        # is successful arrival within 50 m but continue fly until deviating
        if diff is None:
            lon_diff = (interp_lon[-1] - interp_lon[0]) * 111000 * \
                       math.cos(interp_lat[0] / 180 * math.pi)
            lat_diff = (interp_lat[-1] - interp_lat[0]) * 111000
            # compute RD on the premise of not deviating from the prescribed route
            diff = math.sqrt(lon_diff * lon_diff + lat_diff * lat_diff)
            all_diff += diff
            total_num += 1

    print(all_diff // total_num)
    # compute MRD on the premise of not deviating from the prescribed route
    return all_diff // total_num


def get_infer_time(path, a, b):
    """
    compute inference time of a method
    :return: inference time
    """
    noise_db = NOISE_DB

    all_noise_res = []
    total_time = 0
    total_num = 0

    # get all kinds of noises
    for i in range(a, b):
        # get the result txt of a kind of noise
        one_noise_res, _ = get_one_noise_res(path=path,
                                          noise=noise_db[i][0] + "_" + noise_db[i][1])
        all_noise_res.extend(one_noise_res)

    for i in range(0, len(all_noise_res)):
        total_time += float(all_noise_res[i][-1])
        total_num += 1

    print(total_time // total_num)


if __name__ == "__main__":
    path = "../test_our_model"
    a = 0
    b = 1
    # a = 1
    # b = len(NOISE_DB)
    print("SR@25, SR@50:")
    get_success_rate(path, a, b)
    print("MEPE:")
    get_MEPE(path, a, b)
    print("MRE:")
    get_MRE(path, a, b)
    print("MRD:")
    get_MRD(path, a, b)
    print("Inference Time:")
    get_infer_time(path, a, b)
