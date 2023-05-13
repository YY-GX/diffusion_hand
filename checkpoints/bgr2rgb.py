import cv2
import numpy as np
from PIL import Image


for i in range(7):
    file_name = 'real_6_channels/sample-{}-123.png'.format(str(i * 100 + 50))
    dst_name = 'real_6_channels/sample-{}-123_rgb.png'.format(str(i * 100 + 50))
    im_cv = cv2.imread(file_name)
    im_rgb = im_cv.copy()[:, :, ::-1]
    cv2.imwrite(dst_name, im_rgb)

