import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt

from ex1_utils import rotate_image, show_flow
from of_methods import lucas_kanade, horn_schunck


def compute_of(img1, img2, pltTitle):
    U_lk, V_lk = lucas_kanade(img1, img2, 10)

    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.imshow(img1)
    ax1.title.set_text('Frame t')
    ax2.imshow(img2)
    ax2.title.set_text('Frame t+1')
    ax3.title.set_text('Lucas-Kanade')
    # show_flow(U_lk, V_lk, ax1_21, type="angle")
    show_flow(U_lk, V_lk, ax3, type="field", set_aspect=True)
    # fig1.suptitle("Lucas-Kanade Optical Flow")

    U_hs, V_hs = horn_schunck(img1, img2, 500, 0.5)

    # fig2, ((ax2_11, ax2_12), (ax2_21, ax2_22)) = plt.subplots(2, 2)
    ax4.title.set_text('Horn-Schunck')
    # show_flow(U_hs, V_hs, ax2_21, type="angle")
    show_flow(U_hs, V_hs, ax4, type="field", set_aspect=True)
    # fig2.suptitle("'Horn−Schunck Optical Flow")

    plt.tight_layout()
    plt.savefig(pltTitle, bbox_inches='tight')
    plt.show()


random1 = np.random.rand(200, 200).astype(np.float32)
random2 = rotate_image(random1.copy(), -1)

waffle1 = cv2.imread('data/waffle1.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)
waffle2 = cv2.imread('data/waffle2.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)

col1 = cv2.imread('data/00000087.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)
col2 = cv2.imread('data/00000088.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)

lab1 = cv2.imread('data/007.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)
lab2 = cv2.imread('data/008.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)

waffle1fast = cv2.imread('data/waffle1fast.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)
waffle2fast = cv2.imread('data/waffle2fast.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)

# compute_of(random1, random2, 'random')
# compute_of(waffle1, waffle2, 'data/waffle.png')
# compute_of(col1, col2, 'data/collision.png')
compute_of(lab1, lab2, 'data/lab.png')
# compute_of(waffle1fast, waffle2fast, 'data/waffleFast.png')
