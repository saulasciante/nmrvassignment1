import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
import datetime

from ex1_utils import rotate_image, show_flow
from of_methods import lucas_kanade, horn_schunck


def compute_of(img1, img2, pltTitle, normalizeImages=True):

    U_lk, V_lk = lucas_kanade(img1, img2, 10)

    if normalizeImages:
        img1 = img1 / 255.0
        img2 = img2 / 255.0

    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.imshow(img1)
    ax1.title.set_text('Frame t')
    ax2.imshow(img2)
    ax2.title.set_text('Frame t+1')
    ax3.title.set_text('Lucas-Kanade')
    # show_flow(U_lk, V_lk, ax1_21, type="angle")
    show_flow(U_lk, V_lk, ax3, type="field", set_aspect=True)
    # fig1.suptitle("Lucas-Kanade Optical Flow")

    U_hs, V_hs = horn_schunck(img1, img2, 100000, 0.5)

    # fig2, ((ax2_11, ax2_12), (ax2_21, ax2_22)) = plt.subplots(2, 2)
    ax4.title.set_text('Horn-Schunck')
    # show_flow(U_hs, V_hs, ax2_21, type="angle")
    show_flow(U_hs, V_hs, ax4, type="field", set_aspect=True)
    # fig2.suptitle("'Horn−Schunck Optical Flow")

    plt.tight_layout()
    plt.savefig(pltTitle, bbox_inches='tight')
    plt.show()


def lk_parameter_comparison(img1, img2):
    U_lk1, V_lk1 = lucas_kanade(img1, img2, 3)
    U_lk2, V_lk2 = lucas_kanade(img1, img2, 10)
    U_lk3, V_lk3 = lucas_kanade(img1, img2, 15)

    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3))
    ax1.title.set_text('Kernel size = 3x3')
    ax2.title.set_text('Kernel size = 10x10')
    ax3.title.set_text('Kernel size = 15x15')

    show_flow(U_lk1, V_lk1, ax1, type="field", set_aspect=True)
    show_flow(U_lk2, V_lk2, ax2, type="field", set_aspect=True)
    show_flow(U_lk3, V_lk3, ax3, type="field", set_aspect=True)

    plt.tight_layout()
    plt.savefig('data/lkparameters.png', bbox_inches='tight')
    plt.show()


def hs_parameter_comparison(img1, img2):
    img1 = img1 / 255.0
    img2 = img2 / 255.0

    U_hs1, V_hs1 = horn_schunck(img1, img2, 50, 0.5)
    U_hs2, V_hs2 = horn_schunck(img1, img2, 150, 0.5)
    U_hs3, V_hs3 = horn_schunck(img1, img2, 1000, 0.5)

    U_hs4, V_hs4 = horn_schunck(img1, img2, 300, 0.1)
    U_hs5, V_hs5 = horn_schunck(img1, img2, 300, 1)
    U_hs6, V_hs6 = horn_schunck(img1, img2, 300, 3)

    fig1, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12, 7))

    ax1.title.set_text('50 iterations, alpha = 0.5')
    ax2.title.set_text('150 iterations, alpha = 0.5')
    ax3.title.set_text('1000 iterations, alpha = 0.5')
    ax4.title.set_text('Alpha = 0.1, iter = 300')
    ax5.title.set_text('Alpha = 1, iter = 300')
    ax6.title.set_text('Alpha = 3, iter = 300')

    show_flow(U_hs1, V_hs1, ax1, type="field", set_aspect=True)
    show_flow(U_hs2, V_hs2, ax2, type="field", set_aspect=True)
    show_flow(U_hs3, V_hs3, ax3, type="field", set_aspect=True)
    show_flow(U_hs4, V_hs4, ax4, type="field", set_aspect=True)
    show_flow(U_hs5, V_hs5, ax5, type="field", set_aspect=True)
    show_flow(U_hs6, V_hs6, ax6, type="field", set_aspect=True)

    plt.tight_layout()
    plt.savefig('data/hsparameters.png', bbox_inches='tight')
    plt.show()


def time_comparison(img1, img2):

    lkKernelSizes = [3, 10, 15]
    hsIterations = [50, 150, 1000]

    for ks in lkKernelSizes:
        startTime = datetime.datetime.now()
        lucas_kanade(img1, img2, ks)
        endTime = datetime.datetime.now()
        time_diff = (endTime - startTime)
        execution_time = time_diff.total_seconds() * 1000
        print("Lucas Kanade", ks, execution_time)

    for hsit in hsIterations:
        startTime = datetime.datetime.now()
        horn_schunck(img1, img2, hsit, 0.5)
        endTime = datetime.datetime.now()
        time_diff = (endTime - startTime)
        execution_time = time_diff.total_seconds() * 1000
        print("Horn Schunck", hsit, execution_time)


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

# compute_of(random1, random2, 'data/random.png', False)
# compute_of(waffle1, waffle2, 'data/waffle.png')
compute_of(col1, col2, 'data/collision.png')
# compute_of(lab1, lab2, 'data/lab.png')
# compute_of(waffle1fast, waffle2fast, 'data/waffleFast.png')
#
# lk_parameter_comparison(waffle1, waffle2)
# hs_parameter_comparison(waffle1, waffle2)

# time_comparison(waffle1, waffle2)
# time_comparison(lab1, lab2)