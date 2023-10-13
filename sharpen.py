from all_import import *

sharpen = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
), dtype="int")

img = cv2.imread('imgin/pxfuel.jpg')
img_RGB = img
img_RGB = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)

for i in range(img_RGB.shape[2]):
    img_RGB[:, :, i] = signal.convolve2d(img_RGB[:, :, i], sharpen, mode="same", boundary="fill", fillvalue=0)
img_RGB = cv2.cvtColor(src=img_RGB, code=cv2.COLOR_RGB2BGR)

cv2.imwrite('imgout/sharpen.png', img_RGB)
cv2.imshow('sharpen.png', img_RGB)
cv2.waitKey(0)
cv2.destroyAllWindows()