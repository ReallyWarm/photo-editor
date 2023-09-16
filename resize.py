import cv2
import numpy as np

w = int(input("Enter Width : "))
h = int(input("Enter Height : "))

img = cv2.imread('imgin/fox.jpg')
resized_img = cv2.resize(img, (w, h))

cv2.imwrite('imgout/resized.png', resized_img)
cv2.imshow('resized.png', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()