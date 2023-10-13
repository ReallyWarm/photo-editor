import matplotlib.pyplot as plt
import numpy as np
import cv2

def color_threshold(img_inp):

    # get input about user lower and upper thredhold   (RGB)
    lower_threshold = [int(x) for x in input("Enter LOWER Threshold (RGB) ex. xxx xxx xxx : ").split(' ')]
    upper_threshold = [int(x) for x in input("Enter UPPER Threshold (RGB) ex. xxx xxx xxx : ").split(' ')]

    lower = (lower_threshold[0], lower_threshold[1], lower_threshold[2])
    higher = (upper_threshold[1], upper_threshold[1], upper_threshold[2])



    img_final = img_inp.copy()

    # make mask from colot threshold  
    mask = cv2.inRange(img_inp, lower, higher)

    # convert from GRAT TO RGB because it will be mask that have only black and wise
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)


    # make img to HSV beceasue it can change color more natural   ( tint )
    img_final_hsv = cv2.cvtColor(img_final, cv2.COLOR_RGB2HSV)

    user_desire_color = int(input("input color that you want (HUE)  0 - 255 : "))

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
                if mask[y, x][0] == 255 and mask[y, x][1] == 255 and mask[y, x][2] == 255:
                    img_final_hsv[y, x][0] = user_desire_color

    img_final_hsv_rgb = cv2.cvtColor(img_final_hsv, cv2.COLOR_HSV2RGB)


    return img_final_hsv_rgb











    









