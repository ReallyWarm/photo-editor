import cv2
import numpy as np
from scipy.signal import convolve2d

def display(img, name='win'):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# https://stackoverflow.com/questions/62855718/why-would-cv2-color-rgb2gray-and-cv2-color-bgr2gray-give-different-results
def bgr2grey(img):
    img = (img[:,:,0]*0.299) + (img[:,:,1]*0.587) + (img[:,:,2]*0.114)
    img = img.astype(np.uint8)
    return img

# https://stackoverflow.com/questions/53235638/how-should-i-convert-a-float32-image-to-an-uint8-image
def normalize8(img):
    mn = img.min()
    mx = img.max() 
    mx -= mn 
    img = ((img - mn)/mx) * 255
    return np.round(img).astype(np.uint8)

if __name__ == '__main__':
    img = cv2.imread('imgin/fox.jpg')
    grey_img = np.copy(img)

    # flat = np.copy(img)
    # flat = flat.flatten('c')
    # print(flat)

    # print(grey_img.shape)
    grey_img = bgr2grey(grey_img)
    # cv2.imwrite('grey_img.png', grey_img)

    # https://stackoverflow.com/questions/39035510/python-implementing-sobel-operators-with-python-without-opencv
    # a1 = np.matrix([1, 2, 1])
    # a2 = np.matrix([-1, 0, 1])
    # sobelx = a1.T * a2
    # sobely = a2.T * a1

    # https://docs.opencv.org/4.8.0/d2/d2c/tutorial_sobel_derivatives.html
    sobelx = np.matrix([[-1, 0, 1], 
                        [-2, 0, 2], 
                        [-1, 0, 1]])
    
    sobely = np.matrix([[ 1, 2, 1], 
                        [ 0, 0, 0], 
                        [-1,-2,-1]])
    
    Gx = convolve2d(grey_img, sobelx, "same", "symm")
    Gy = convolve2d(grey_img, sobely, "same", "symm")
    G = np.sqrt(Gx**2 + Gy**2)
    
    
    a = normalize8(G) # This function convert an image from single precision floating point (i.e. float32) to uint8
    b = cv2.convertScaleAbs(G, alpha=1, beta=0) # This function converts FP32 (single precision floating point) from/to FP16 (half precision floating point)
    # b = cv2.convertScaleAbs(G, alpha=255/G.max()) # same as normalize8()

    # print(G)
    # print(a)
    # print(b)

# ----- TEST SOBEL ----- #
    # mat = np.matrix([[ 5,  5,  5,  5,  5, 5], 
    #                  [ 5,  8, 10, 10,  8, 5],  
    #                  [ 5, 10, 20, 20, 10, 5],
    #                  [ 5, 10, 20, 20, 10, 5],
    #                  [ 5,  8, 10, 10,  8, 5],
    #                  [ 5,  5,  5,  5,  5, 5]])
    
    # mx = convolve2d(mat, sobelx, "same", "symm")
    # my = convolve2d(mat, sobely, "same", "symm")
    # mm = np.sqrt(mx**2 + my**2)
    # mm = mm.astype(np.uint8)

    # print(mx)
    # print()
    # print(my)
    # print()
    # print(mm)
# /---- TEST SOBEL ----/ #

    # display(img)
    # display(x)
    # display(grey_img)
    # display(y)
    # display(G)
    display(a)
    display(b)

    cv2.imwrite('imgout/IMAGE1.png', G)