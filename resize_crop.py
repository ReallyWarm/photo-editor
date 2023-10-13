import cv2

def crop_tools(image, x, y, width, height):
    return image[y:y+height, x:x+width]

def resizing(img):
    width, height = map(int, input("Enter Width & Height to resize (e.g., 640 480): ").split())
    resized_img = cv2.resize(img, (width, height))

    output_resize_path = "Resized folder/image01.jpg"
    cv2.imwrite(output_resize_path, resized_img)
    print(f"Resized image saved at: {output_resize_path}")
    cv2.imshow('Resized Image', resized_img)

def croping(img):
    x, y, crop_width, crop_height = map(int, input("Enter X, Y, Width, and Height to crop (e.g., 100 100 400 300): ").split())
    cropped_img = crop_tools(img, x, y, crop_width, crop_height)
    return cropped_img
