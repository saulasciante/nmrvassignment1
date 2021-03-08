import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt

from ex1_utils import rotate_image, show_flow
from of_methods import lucas_kanade, horn_schunck


# img1 = cv2.imread('disparity/cporta_left.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
# img2 = cv2.imread('disparity/cporta_right.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
img1 = np.random.rand(200, 200).astype(np.float32)
img2 = img1.copy()
img2 = rotate_image(img2, -1)

# U_lk, V_lk = lucas_kanade(img1, img2, 3)
U_hs, V_hs = horn_schunck(img1, img2, 1000, 0.5)

# fig1, ((ax1_11, ax1_12), (ax1_21, ax1_22)) = plt.subplots(2, 2)
# ax1_11.imshow(img1)
# ax1_12.imshow(img2)
# show_flow(U_lk, V_lk, ax1_21, type="angle")
# show_flow(U_lk, V_lk, ax1_22, type="field", set_aspect=True)
# fig1.suptitle("Lucas-Kanade Optical Flow")

fig2, ((ax2_11, ax2_12), (ax2_21, ax2_22)) = plt.subplots(2, 2)
ax2_11.imshow(img1)
ax2_12.imshow(img2)
show_flow(U_hs, V_hs, ax2_21, type="angle")
show_flow(U_hs, V_hs, ax2_22, type="field", set_aspect=True)
fig2.suptitle("'Horn−Schunck Optical Flow")

plt.show()