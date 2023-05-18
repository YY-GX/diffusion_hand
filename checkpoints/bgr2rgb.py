import cv2
import numpy as np
from PIL import Image

folder = "synth_6_channels_left_right_with_bg"
#folder = "synth_6_channels_left_right_no_bg"

# for i in range(7):
#     file_name = '{}/sample-{}-123.png'.format(folder, str(i * 100 + 50))
#     dst_name = '{}/sample-{}-123_rgb.png'.format(folder, str(i * 100 + 50))
for i in [290]:
    file_name = '{}/sample-{}-123.png'.format(folder, i)
    dst_name = '{}/sample-{}-123_rgb.png'.format(folder, i)
    im_cv = cv2.imread(file_name)
    im_rgb = im_cv.copy()[:, :, ::-1]
    cv2.imwrite(dst_name, im_rgb)

