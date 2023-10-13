import cv2

def resize_image(img):
    w, h = input("Enter width and height (Ex.100 100) : ").split()
    resized_img = cv2.resize(img, (int(w), int(h)), interpolation=1)
    return resized_img

def crop_tools(image, x, y, width, height):
    return image[y:y+height, x:x+width]

def croping(img):
    x, y, crop_width, crop_height = map(int, input("Enter X, Y, Width, and Height to crop (e.g., 100 100 400 300): ").split())
    cropped_img = crop_tools(img, x, y, crop_width, crop_height)
    return cropped_img
