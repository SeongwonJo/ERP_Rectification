import numpy as np
import cv2
import argparse


# def arguments_setting():
#     arguments = argparse.ArgumentParser()
#     arguments.add_argument('-i', '--image_path', default="./erp.png")
#     arguments.add_argument('--vfov', type=int, default=60)
#     arguments.add_argument('--hfov', type=int, default=120)
#     args = arguments.parse_args()
#     return args


def erp2rect(img, vfov, hfov):
    src_height, src_width = img.shape[:2]
    f = src_width / (2 * np.pi) # focal length

    vfov = np.deg2rad(vfov)
    hfov = np.deg2rad(hfov)

    w_prime = int(2 * f * np.tan(hfov/2))
    h_prime = int(2 * f * np.tan(vfov/2))

    dst = np.zeros((h_prime, w_prime, 3), dtype=np.uint8)

    cx_prime = w_prime / 2
    cy_prime = h_prime / 2

    # topview_projection = np.zeros((src_height, src_width, 3), dtype=np.uint8)
    # cx = (src_width - 1) / 2
    # cy = (src_height - 1) / 2
    #
    # for y in range(src_height):
    #     phi = (y - cy) / f
    #     D = f / np.tan(phi)
    #
    #     for x in range(src_width):
    #         theta = (x - cx) / f
    #         tvp_y = int(cy - D * np.cos(theta))
    #         tvp_x = int(cx + D * np.sin(theta))
    #         # print(f'[{tvp_y},{tvp_x}]')
    #
    #         topview_projection[y, x, :] = img[tvp_y, tvp_x, :]

    for x in range(w_prime):
        x_th = np.arctan((x - cx_prime) / f) # np.arctan 출력은 [-pi/2, pi/2] radian 값
        src_x = int((src_width / np.deg2rad(360)) * (x_th + np.deg2rad(180)))

        y_f = f / np.cos(x_th)

        for y in range(h_prime):
            y_th = np.arctan((y - cy_prime) / y_f)
            src_y = int((src_height / np.deg2rad(180)) * (y_th + np.deg2rad(90)))

            dst[y, x, :] = img[src_y, src_x, :]

    return dst


if __name__ == "__main__":
    # args = arguments_setting()
    img_path = 'erp.png'
    vfov = 170
    hfov = 170

    img = cv2.imread(img_path)

    result = erp2rect(img, vfov, hfov)

    cv2.imshow('test', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("result.png", result)
