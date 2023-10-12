import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np
np.set_printoptions(threshold=np.inf)
import cv2

# flags = [i for i in dir(cv2) if i.startswith("COLOR_")]    
# print(flags)


nemo = cv2.imread("images/nemo0.jpg")

nemo_rgb = cv2.cvtColor(nemo, cv2.COLOR_BGR2RGB)
nemo_hsv = cv2.cvtColor(nemo, cv2.COLOR_BGR2HSV)


low_white = np.uint8([[[210, 209, 209]]])
high_white = np.uint8([[[255, 255, 255]]])

hsv_low_white = cv2.cvtColor(low_white, cv2.COLOR_RGB2HSV)
hsv_high_white = cv2.cvtColor(high_white, cv2.COLOR_RGB2HSV)



light_orange = (1, 190, 200)

dark_orange = (18, 255, 255)


mask = cv2.inRange(nemo_hsv, hsv_low_white, hsv_high_white)

cv2.imshow("mask", mask)



indice = np.where(mask == 255)

nemo_rgb[indice[0], indice[1], :] = [0, 255, 0]


cv2.waitKey(0)






