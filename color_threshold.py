from all_import import *

def color_threshold(img_inp):

    # get input about user lower and upper thredhold   (RGB)
    hue_threshold = [int(x) for x in input("(HSV) Enter range HUE Threshold  < 0-359 deg > ex. 120-300 : ").split('-')]
    sat_threshold = [int(x) for x in input("(HSV) Enter range Saturation Threshold  < 0-100 % > ex. 30-80 : ").split('-')]
    value_threshold = [int(x) for x in input("(HSV) Enter range Value Threshold < 0-100 % > ex. 50-90 : ").split('-')]

    lower = (np.interp(hue_threshold[0], [0, 365], [0, 180]), np.interp(sat_threshold[0], [0, 100], [0, 255]), np.interp(value_threshold[0], [0, 100], [0, 255]))
    higher = (np.interp(hue_threshold[1], [0, 365], [0, 180]), np.interp(sat_threshold[1], [0, 100], [0, 255]), np.interp(value_threshold[1], [0, 100], [0, 255]))

    img_final = img_inp.copy()
    img_inp_hsv = cv2.cvtColor(img_inp, cv2.COLOR_RGB2HSV)


    # make mask from colot threshold  
    mask = cv2.inRange(img_inp_hsv, lower, higher)

    # convert from GRAT TO RGB because it will be mask that have only black and wise
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)


    color_layer = img_inp.copy()
    desire_color_RGB = [int(x) for x in input("input color that you want (RGB) xxx xxx xxx : ").split(' ')]
    beta = int(input("Weight (0 - 100%) ex. 10 : "))
    beta = beta / 100
    alpha = 1 - beta


    color_layer[(mask == 255).all(-1)] = desire_color_RGB

    cv2.addWeighted(img_final, alpha, color_layer, beta, 0, img_final)



    # # make img to HSV beceasue it can change color more natural   ( tint )
    # img_final_hsv = cv2.cvtColor(img_final, cv2.COLOR_RGB2HSV)

    # user_desire_color = int(input("input color that you want (HUE)  0 - 255 : "))

    # for y in range(mask.shape[0]):
    #     for x in range(mask.shape[1]):
    #             if mask[y, x][0] == 255 and mask[y, x][1] == 255 and mask[y, x][2] == 255:
    #                 img_final_hsv[y, x][0] = user_desire_color

    # img_final_hsv_rgb = cv2.cvtColor(img_final_hsv, cv2.COLOR_HSV2RGB)

    return img_final











    









