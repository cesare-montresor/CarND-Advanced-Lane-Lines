import cv2
import os
import numpy as np

import matplotlib.pyplot as plt
import time



test_image_original = cv2.imread( './test_images/straight_lines1.jpg')
test_image_original = cv2.cvtColor(test_image_original,cv2.COLOR_BGR2RGB)
plt.imshow(test_image_original, cmap="gray")
plt.show()
plt.close('all')

test_image_original = cv2.imread( './test_images/straight_lines2.jpg')
test_image_original = cv2.cvtColor(test_image_original,cv2.COLOR_BGR2RGB)
plt.imshow(test_image_original, cmap="gray")
plt.show()
plt.close('all')

