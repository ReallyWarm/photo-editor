from all_import import *

def resize_image(img):
    w, h = input("Enter width and height (Ex.100 100) : ").split()
    resized_img = cv2.resize(img, (int(w), int(h)), interpolation=1)
    return resized_img