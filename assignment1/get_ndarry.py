import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

cv2.startWindowThread()

# 读入图片
img = cv2.imread("./input.jpeg", 0)
cv2.imwrite("input_grey.jpeg", img)

# 显示图片
# cv2.imshow("image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

img = Image.open("input_grey.jpeg", mode="r")
tensor = np.array(img)
np.save("input.npy", np.array(tensor))