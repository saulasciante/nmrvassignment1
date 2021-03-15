import numpy as np

from ex1_utils import gaussderiv, convolve, gausssmooth
from sklearn.metrics.pairwise import cosine_similarity

# ignore if division with zero
np.seterr(divide='ignore', invalid='ignore')


def discetize_derivatives(img1, img2, smoothingSigma, derivSigma):
    Ix1, Iy1 = gaussderiv(img1, derivSigma)
    Ix2, Iy2 = gaussderiv(img2, derivSigma)
    Ix, Iy = np.mean([Ix1, Ix2], axis=0), np.mean([Iy1, Iy2], axis=0)
    It = gausssmooth(img2 - img1, smoothingSigma)

    return Ix, Iy, It


def lucas_kanade(img1, img2, N):
    Ix, Iy, It = discetize_derivatives(img1, img2, 1, 0.4)

    kernel = np.ones((N, N))

    sum_Iy_squared = convolve(np.square(Iy), kernel)
    sum_Ix_squared = convolve(np.square(Ix), kernel)

    sum_Ix_Iy = convolve(np.multiply(Ix, Iy), kernel)

    sum_Ix_It = convolve(np.multiply(Ix, It), kernel)
    sum_Iy_It = convolve(np.multiply(Iy, It), kernel)

    D = np.subtract(
        np.multiply(sum_Ix_squared, sum_Iy_squared),
        np.square(sum_Ix_Iy)
    )

    D[D == 0] = 0.000000001

    delta_x = np.divide(
        np.add(
            np.multiply(-sum_Iy_squared, sum_Ix_It),
            np.multiply(sum_Ix_Iy, sum_Iy_It)
        ),
        D
    )

    delta_y = np.divide(
        np.subtract(
            np.multiply(sum_Ix_Iy, sum_Ix_It),
            np.multiply(sum_Ix_squared, sum_Iy_It)
        ),
        D
    )

    return delta_x, delta_y


def horn_schunck(img1, img2, n_iters, lmbd, lkInit=False):
    Ix, Iy, It = discetize_derivatives(img1, img2, 1, 1)

    #  speed by lukas kanade
    if lkInit:
        u, v = lucas_kanade(img1, img2, 10)
    else:
        u = np.zeros((Ix.shape[0], Ix.shape[1]))
        v = np.zeros((Iy.shape[0], Iy.shape[1]))

    u_avg = np.ones((Ix.shape[0], Ix.shape[1]))
    v_avg = np.ones((Iy.shape[0], Iy.shape[1]))

    u_sim = 0  # similarity between u and u_avg
    v_sim = 0  # similarity between v and v_avg

    kernel = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]])

    D = np.square(Ix) + np.square(Iy) + lmbd

    matrixElements = Ix.shape[0] * Ix.shape[1]
    i = 0

    while i < n_iters and u_sim < 0.6 and v_sim < 0.6:

        u_sim = round(np.sum(cosine_similarity(u, u_avg)) / matrixElements, 4)
        v_sim = round(np.sum(cosine_similarity(v, v_avg)) / matrixElements, 4)

        if i % 100 == 0:
            print(i, u_sim, v_sim)

        u_avg = convolve(u, kernel)
        v_avg = convolve(v, kernel)

        P = np.divide(
            sum([
                It,
                np.multiply(Ix, u_avg),
                np.multiply(Iy, v_avg)
            ]),
            D
        )

        u = np.subtract(u_avg, np.multiply(Ix, P))
        v = np.subtract(v_avg, np.multiply(Iy, P))

        i = i + 1

    print(i)

    return u, v










