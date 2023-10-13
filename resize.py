from all_import import *

def resize_image(img):
    w, h = input("Enter width and height : ").split()
    resized_img = cv2.resize(img, (w, h))
    return resized_img