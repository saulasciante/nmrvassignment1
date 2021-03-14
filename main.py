import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt

from ex1_utils import rotate_image, show_flow
from of_methods import lucas_kanade, horn_schunck


def compute_of(img1, img2):
    U_lk, V_lk = lucas_kanade(img1, img2, 10)

    fig1, ((ax1_11, ax1_12), (ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(img1)
    ax1_12.imshow(img2)
    show_flow(U_lk, V_lk, ax1_21, type="angle")
    show_flow(U_lk, V_lk, ax1_22, type="field", set_aspect=True)
    fig1.suptitle("Lucas-Kanade Optical Flow")

    U_hs, V_hs = horn_schunck(img1, img2, 2000, 0.4)

    fig2, ((ax2_11, ax2_12), (ax2_21, ax2_22)) = plt.subplots(2, 2)
    ax2_11.imshow(img1)
    ax2_12.imshow(img2)
    show_flow(U_hs, V_hs, ax2_21, type="angle")
    show_flow(U_hs, V_hs, ax2_22, type="field", set_aspect=True)
    fig2.suptitle("'Horn−Schunck Optical Flow")

    plt.show()


waffle1 = cv2.imread('data/waffle1.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)
waffle2 = cv2.imread('data/waffle2.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)

col1 = cv2.imread('data/collision/00000087.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)
col2 = cv2.imread('data/collision/00000088.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)

lab1 = cv2.imread('data/lab2/007.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)
lab2 = cv2.imread('data/lab2/008.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)

# img1 = np.random.rand(200, 200).astype(np.float32)
# img2 = img1.copy()
# img2 = rotate_image(img2, -1)

compute_of(waffle1, waffle2)
compute_of(col1, col2)
compute_of(lab1, lab2)
