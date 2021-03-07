import cv2
import numpy as np

from ex1_utils import gaussderiv, convolve

sigma = 0.4

# ignore division with zero
np.seterr(divide='ignore', invalid='ignore')


def lucas_kanade(img1, img2, N):
    Ix, Iy = gaussderiv(img1, sigma)
    It = img2 - img1

    sum_Iy_squared = convolve(np.square(Iy), N)
    sum_Ix_squared = convolve(np.square(Ix), N)

    sum_Ix_Iy = convolve(np.multiply(Ix, Iy), N)

    sum_Ix_It = convolve(np.multiply(Ix, It), N)
    sum_Iy_It = convolve(np.multiply(Iy, It), N)

    delta_x = np.divide(
        np.add(
            np.multiply(-sum_Iy_squared, sum_Ix_It),
            np.multiply(sum_Ix_Iy, sum_Iy_It)
        ),
        np.subtract(
            np.multiply(sum_Ix_squared, sum_Iy_squared),
            np.square(sum_Ix_Iy)
        )
    )

    delta_y = np.divide(
        np.subtract(
            np.multiply(sum_Ix_Iy, sum_Ix_It),
            np.multiply(sum_Ix_squared, sum_Iy_It)
        ),
        np.subtract(
            np.multiply(sum_Ix_squared, sum_Iy_squared),
            np.square(sum_Ix_Iy)
        )
    )

    return delta_x, delta_y


def horn_schunck(img1, img2, n_iters, lmbd):
    return


# waffleImg = cv2.cvtColor(cv2.imread('waffle.jpg'), cv2.COLOR_RGB2GRAY)
# testImg1 = cv2.cvtColor(cv2.imread('lab2/001.jpg'), cv2.COLOR_RGB2GRAY)
# testImg2 = cv2.cvtColor(cv2.imread('lab2/002.jpg'), cv2.COLOR_RGB2GRAY)
# lucas_kanade(testImg1, testImg2, 3)
