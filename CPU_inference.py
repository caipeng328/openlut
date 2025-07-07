import sys
sys.path.append(".")
from PyApplyLUT import PyApplyLUT
from lut_tools import cube_to_npy

import cv2
import numpy as np
from pathlib2 import Path
import time


LUT_FILE = Path("./test/1.cube")

img = cv2.imread("./test/1.jpg")
img = img / 255

alut = PyApplyLUT(lut_file=LUT_FILE)
t = time.time()
new_img = alut.apply_lut(img)
print(time.time() - t)

new_img = new_img * 255
cv2.imwrite("./test/new_img_1.jpg",new_img)

