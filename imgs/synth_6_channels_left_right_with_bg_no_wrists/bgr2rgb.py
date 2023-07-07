import os
import cv2
import numpy as np
img_ls = os.listdir()
for img_pth in img_ls:
    if '456' in img_pth or '.png' not in img_pth:
        continue
    im_bgr = cv2.imread(img_pth)
    im_rgb = im_bgr[:, :, ::-1]
    cv2.imwrite(img_pth, im_rgb)
