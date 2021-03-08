import numpy as np

from ex1_utils import gaussderiv, convolve, gausssmooth

derivSigma = 0.4

# ignore if division with zero
np.seterr(divide='ignore', invalid='ignore')


def discetize_derivatives(img1, img2, smoothingSigma):
    img1 = gausssmooth(img1, smoothingSigma)
    img2 = gausssmooth(img2, smoothingSigma)

    Ix1, Iy1 = gaussderiv(img1, derivSigma)
    Ix2, Iy2 = gaussderiv(img2, derivSigma)
    Ix, Iy = np.mean([Ix1, Ix2], axis=0), np.mean([Iy1, Iy2], axis=0)
    It = img2 - img1

    return Ix, Iy, It


def lucas_kanade(img1, img2, N):
    Ix, Iy, It = discetize_derivatives(img1, img2, 1)

    kernel = np.ones((N, N))

    sum_Iy_squared = convolve(np.square(Iy), kernel)
    sum_Ix_squared = convolve(np.square(Ix), kernel)

    sum_Ix_Iy = convolve(np.multiply(Ix, Iy), kernel)

    sum_Ix_It = convolve(np.multiply(Ix, It), kernel)
    sum_Iy_It = convolve(np.multiply(Iy, It), kernel)

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
    Ix, Iy, It = discetize_derivatives(img1, img2, 1)

    u = np.zeros((Ix.shape[0], Ix.shape[1]))
    v = np.zeros((Iy.shape[0], Iy.shape[1]))

    kernel = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]])

    for i in range(n_iters):
        u_avg = convolve(u, kernel)
        v_avg = convolve(v, kernel)

        P = np.divide(
            sum([
                It,
                np.multiply(Ix, u_avg),
                np.multiply(Iy, v_avg)
            ]),
            sum([
                np.square(Ix),
                np.square(Iy),
                lmbd
            ])
        )

        u = np.subtract(u_avg, np.multiply(Ix, P))
        v = np.subtract(v_avg, np.multiply(Iy, P))

    return u, v










