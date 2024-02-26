import os
from io import BytesIO
import cv2
import numpy as np


def pic_compress(pic_path, out_path, target_size=199, quality=90, step=5, pic_type='.jpg'):
    # read images bytes
    with open(pic_path, 'rb') as f:
        pic_byte = f.read()

    img_np = np.frombuffer(pic_byte, np.uint8)
    img_cv = cv2.imdecode(img_np, cv2.IMREAD_ANYCOLOR)

    current_size = len(pic_byte) / 1024
    print("image size before compression (KB): ", current_size)
    while current_size > target_size:
        pic_byte = cv2.imencode(pic_type, img_cv, [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1]
        if quality - step < 0:
            break
        quality -= step
        current_size = len(pic_byte) / 1024

    # save image
    with open(out_path, 'wb') as f:
        f.write(BytesIO(pic_byte).getvalue())

    return len(pic_byte) / 1024


def process_order():
    # ori,random,uni,hei 1
    # cutout,rain,snow,fog,bright 5
    # dir = "../../sequence/processOrder/order/74551"
    # b = "/order"
    # c = "/74551"
    # dir = "../../sequence/processOrder/order/97000"
    # b = "/order"
    # c = "/97000"
    # dir = "../../sequence/processOrder/order/97001"
    # b = "/order"
    # c = "/97001"
    # dir = "../../sequence/processOrder/order/97002"
    # b = "/order"
    # c = "/97002"
    # dir = "../../sequence/processOrder/order/97003"
    # b = "/order"
    # c = "/97003"
    dir = "../../sequence/processOrder/order/97004"
    b = "/order"
    c = "/97004"

    new_dir = "../" + "compress" + b
    if os.path.exists(new_dir) is False:
        os.mkdir(new_dir)
    new_dir = "../" + "compress" + b + c
    if os.path.exists(new_dir) is False:
        os.mkdir(new_dir)

    paths = os.listdir(dir)
    paths.sort()

    j = 0
    for pic in paths:
        full_pic_path = os.path.join(dir, pic)
        new_full_pic_path = os.path.join(new_dir, str(j) + ".png")
        pic_size = pic_compress(full_pic_path, new_full_pic_path, target_size=10)
        print("image size after compression (KB): ", pic_size)
        j += 1


def process_our_model():
    # our ori+disturbance 9
    # a = "/test_wohard"
    # b = "/ori_ori"
    # c = "/test1"
    # a = "/test_our_model"
    # b = "/ori_random"
    # c = "/test5"
    # a = "/test_our_model"
    # b = "/ori_uni"
    # c = "/test11"
    # a = "/test_our_model"
    # b = "/ori_hei"
    # c = "/test15"
    # a = "/test_our_model"
    # b = "/cutout_ori"
    # c = "/test7"
    # a = "/test_our_model"
    # b = "/fog_ori"
    # c = "/test10"
    # a = "/test_our_model"
    # b = "/snow_ori"
    # c = "/test6"
    # a = "/test_our_model"
    # b = "/rain_ori"
    # c = "/test1"
    # a = "/test_our_model"
    # b = "/bright_ori"
    # c = "/test2"

    # a = "/test_fsra33"
    # b = "/ori_ori"
    # c = "/test13"
    # a = "/test_fsra33"
    # b = "/ori_random"
    # c = "/test0"
    # a = "/test_fsra33"
    # b = "/ori_uni"
    # c = "/test5"
    # a = "/test_fsra33"
    # b = "/ori_hei"
    # c = "/test6"
    # a = "/test_fsra33"
    # b = "/cutout_ori"
    # c = "/test6"
    # a = "/test_fsra33"
    # b = "/fog_ori"
    # c = "/test0"
    # a = "/test_fsra33"
    # b = "/snow_ori"
    # c = "/test0"
    # a = "/test_fsra33"
    # b = "/rain_ori"
    # c = "/test17"
    # a = "/test_fsra33"
    # b = "/bright_ori"
    # c = "/test0"

    # a = "/test_lpn17"
    # b = "/ori_ori"
    # c = "/test79"
    # a = "/test_lpn9"
    # b = "/ori_random"
    # c = "/test15"
    # a = "/test_lpn9"
    # b = "/ori_uni"
    # c = "/test18"
    # a = "/test_lpn9"
    # b = "/ori_hei"
    # c = "/test16"
    # a = "/test_lpn9"
    # b = "/cutout_ori"
    # c = "/test6"
    # a = "/test_lpn9"
    # b = "/fog_ori"
    # c = "/test0"
    # a = "/test_lpn9"
    # b = "/snow_ori"
    # c = "/test2"
    # a = "/test_lpn9"
    # b = "/rain_ori"
    # c = "/test0"
    a = "/test_lpn9"
    b = "/bright_ori"
    c = "/test6"

    dir = ".." + a + b + c
    new_dir = "../" + "compress" + a
    if os.path.exists(new_dir) is False:
        os.mkdir(new_dir)
    new_dir = "../" + "compress" + a + b
    if os.path.exists(new_dir) is False:
        os.mkdir(new_dir)
    new_dir = "../" + "compress" + a + b + c
    if os.path.exists(new_dir) is False:
        os.mkdir(new_dir)

    paths = os.listdir(dir)
    paths.sort(key=lambda x: int(x[:x.index(',')]))

    j = 0
    for pic in paths:
        full_pic_path = os.path.join(dir, pic)
        new_full_pic_path = os.path.join(new_dir, str(j) + ".png")
        pic_size = pic_compress(full_pic_path, new_full_pic_path, target_size=10)
        print("image size after compression (KB): ", pic_size)
        j += 1


if __name__ == '__main__':
    # process_order()
    process_our_model()
