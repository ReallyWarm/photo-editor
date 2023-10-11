import cv2, numpy as np
from scipy.signal import convolve2d

def display(img, name='win'):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def gaussian_kernel(size, sigma=1):
    if size % 2 != 1:
        return None
    size = size // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2*sigma**2))) * normal
    return g

def sobel_filter(grey_img):
    sobelx = np.matrix([[-1, 0, 1], 
                        [-2, 0, 2], 
                        [-1, 0, 1]], np.float32)
    
    sobely = np.matrix([[ 1, 2, 1], 
                        [ 0, 0, 0], 
                        [-1,-2,-1]], np.float32)
    
    Gx = convolve2d(grey_img, sobelx, "same", "symm")
    Gy = convolve2d(grey_img, sobely, "same", "symm")

    Gout = np.hypot(Gx, Gy) #sqrt(Gx**2 + Gy**2)
    theta = np.arctan2(Gy, Gx)

    return Gout, theta

# https://github.com/UsamaI000/CannyEdgeDetection-from-scratch-python/blob/master/CannyEdgeDetector.ipynb
def gradient_direction(theta):
    angle = np.rad2deg(theta) + 180
    return angle

# https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
def non_max_suppression(grad, angle):
    h, w = grad.shape
    nms = np.zeros((h, w), dtype=np.uint8)
    
    for i in range(1, h-1):
        for j in range(1, w-1):
            q = 255
            r = 255
            
            # angle 0 and 180
            if (0 <= angle[i,j] < 22.5) or (337.5 <= angle[i,j] <= 360) or (157.5 <= angle[i,j] < 202.5):
                q = grad[i,j+1]
                r = grad[i,j-1]
            # angle 45 and 225
            elif (22.5 <= angle[i,j] < 67.5) or (202.5 <= angle[i, j] < 247.5):
                q = grad[i+1,j-1]
                r = grad[i-1,j+1]
            # angle 90 and 270
            elif (67.5 <= angle[i,j] < 112.5) or (247.5 <= angle[i, j] < 292.5):
                q = grad[i+1,j]
                r = grad[i-1,j]
            # angle 135 and 315
            elif (112.5 <= angle[i,j] < 157.5) or (292.5 <= angle[i, j] < 337.5):
                q = grad[i-1,j-1]
                r = grad[i+1,j+1]

            if (grad[i,j] >= q) and (grad[i,j] >= r):
                nms[i,j] = grad[i,j]
            else:
                nms[i,j] = 0
    
    return nms

def double_thresholding(suppressed, low_ratio=0.05, high_ratio=0.09, weak=25, strong=255):
    high_threshold = suppressed.max() * high_ratio
    low_threshold = high_threshold * low_ratio

    h, w = suppressed.shape
    thresholded = np.zeros((h, w), dtype=np.uint8)

    # for i in range(h):
    #     for j in range(w):
    #         if suppressed[i,j] > high_threshold:
    #             thresholded[i,j] = strong
    #         elif suppressed[i,j] >= low_threshold and suppressed[i,j] <= high_threshold:
    #             thresholded[i,j] = weak
    #         else:
    #             thresholded[i,j] = 0

    strong_i, strong_j = np.where(suppressed >= high_threshold)
    weak_i, weak_j = np.where((suppressed <= high_threshold) & (suppressed >= low_threshold))

    thresholded[strong_i, strong_j] = strong
    thresholded[weak_i, weak_j] = weak

    return thresholded, weak, strong

def hysteresis(threshold, weak=25, strong=255):
    h, w = threshold.shape
    strong_edge = np.zeros((h, w), dtype=np.uint8)

    for i in range(1, h-1):
        for j in range(1, w-1):

            if threshold[i,j] == weak:
                if (threshold[i-1,j] == strong) or (threshold[i-1,j+1] == strong) or (threshold[i,j+1] == strong) or (threshold[i+1,j+1] == strong) or \
                   (threshold[i+1,j] == strong) or (threshold[i+1,j-1] == strong) or (threshold[i,j-1] == strong) or (threshold[i-1,j-1] == strong):
                    strong_edge[i,j] = strong

            elif threshold[i,j] == strong:
                strong_edge[i,j] = strong

    return strong_edge

def canny_detection(image, new_gaussian=None, low_threshold=0.05, high_threshold=0.09, weak_mag=25, strong_mag=255):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if new_gaussian is None:
        gaussian_filter = np.matrix([[2,  4,  5,  4, 2],
                                     [4,  9, 12,  9, 4],
                                     [5, 12, 15, 12, 5],
                                     [4,  9, 12,  9, 4],
                                     [2,  4,  5,  4, 2]]) * 1/159
    else:
        gaussian_filter = new_gaussian

    blur = convolve2d(grey, gaussian_filter, 'same', 'symm')

    grad, theta = sobel_filter(blur)
    grad = cv2.convertScaleAbs(grad)

    angle = gradient_direction(theta)
    nms = non_max_suppression(grad, angle)

    thresh, weak, strong = double_thresholding(nms, low_threshold, high_threshold, weak_mag, strong_mag)
    canny = hysteresis(thresh, weak, strong)

    # display(grad)
    # display(cv2.convertScaleAbs(angle))
    # display(nms)
    # display(thresh)
    # display(canny)

    return canny

if __name__ == '__main__':
    img = cv2.imread('imgin/test.png')
    canny = canny_detection(img.copy(), gaussian_kernel(5, sigma=1.6))
    display(canny)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    dilated = cv2.morphologyEx(canny.copy(), cv2.MORPH_CLOSE, kernel, iterations=2)
    # dilated = cv2.dilate(canny.copy(), kernel, iterations=1)
    cnts, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    display(dilated)

    coins = img.copy()
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 100:
            cv2.drawContours(coins , [c], -1, (0, 0, 255), -1)
    display(coins)

    # grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # gaussian_filter = np.matrix([[2,  4,  5,  4, 2],
    #                              [4,  9, 12,  9, 4],
    #                              [5, 12, 15, 12, 5],
    #                              [4,  9, 12,  9, 4],
    #                              [2,  4,  5,  4, 2]]) * 1/159

    # blur = convolve2d(grey, gaussian_filter, 'same', 'symm')
    # # bx = blur.astype(np.uint8)
    # # blur = cv2.convertScaleAbs(blur)
    # # bx = cv2.GaussianBlur(grey, [5,5], 0)

    # # print(gaussian_filter)
    # # print(cv2.getGaussianKernel(5,1.6))
    # # print(gaussian_kernel(5,1.55))

    # grad, theta = sobel_filter(blur)
    # grad = cv2.convertScaleAbs(grad)
    # # bx = normalize8(grad)

    # angle = gradient_direction(theta)
    # nms = non_max_suppression(grad, angle)

    # thresh, weak, strong = double_thresholding(nms)
    # canny = hysteresis(thresh, weak, strong)
    # # print(grad)
    # # print(bx)
    # # display(grey)
    # # display(bx)

    # display(grad)
    # # display(cv2.convertScaleAbs(angle))
    # display(nms)
    # display(thresh)
    # display(canny)
