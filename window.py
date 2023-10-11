import cv2, numpy as np
from canny import canny_detection, gaussian_kernel

def nothing(x):
    pass
# create trackbars for color change
if __name__ == '__main__':
    cv2.namedWindow('Editor')
    load = cv2.imread('imgin/test.png')
    # off = np.zeros(load.shape)
    img = load
    canny = load

    # create switch for ON/OFF functionality
    cv2.createTrackbar('process canny', 'Editor',0,1,nothing)
    switch = '0 : OFF \n1 : ON'
    cv2.createTrackbar(switch, 'Editor',0,1,nothing)

    cv2.createTrackbar('low threshold','Editor',0,200,nothing)
    cv2.createTrackbar('high threshold','Editor',0,200,nothing)

    cv2.createTrackbar('new gaussian:','Editor',0,1,nothing)
    cv2.createTrackbar('gaussian size','Editor',3,15,nothing)
    cv2.createTrackbar('gaussian sigma','Editor',0,400,nothing)

    while(1):
        cv2.imshow('Editor',img)
        k = cv2.waitKey(10) & 0xFF
        if k == 27:
            break

        # get current positions of four trackbars
        lth = cv2.getTrackbarPos('low threshold','Editor')
        hth = cv2.getTrackbarPos('high threshold','Editor')
        ng = cv2.getTrackbarPos('new gaussian:','Editor')
        gsz = cv2.getTrackbarPos('gaussian size','Editor')
        gsg = cv2.getTrackbarPos('gaussian sigma','Editor')

        s = cv2.getTrackbarPos(switch,'Editor')
        p = cv2.getTrackbarPos('process canny','Editor')

        if p == 1:
            g = None
            if ng == 1:
                g = gaussian_kernel(gsz, gsg/100)
            canny = canny_detection(load, g, lth/1000, hth/1000)

        if s == 0:
            img = load
        else:
            img = canny

    cv2.destroyAllWindows()