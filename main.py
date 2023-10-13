#from canny import *
from color_threshold import *
#from npsobel import *
from resize import *
from rotate import *
#from sharpen import *
#from sobel import *
from cos_sim import *

mode_list = ["Resize", "Rotate", "Crop", "Color Threshold", "Edge Detect"]

while True:
    img_name = input("Enter Image Name (Ex. 1.jpg): ")
    try:
        img = cv2.imread(f"imgin/{img_name}")
    except FileNotFoundError as e:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print("Choose Features to Edit Image")
    for i in range(len(mode_list)):
        print(f"[{i + 1}]: {mode_list[i]}")
    mode = int(input("Please Select Feature : "))

    if mode == 1:
        edited_image = resize_image(img)
    elif mode == 2:
        edited_image = rotate_image(img)
    elif mode == 3:
        pass
    elif mode == 4:
        edited_image = color_threshold(img)
    elif mode == 5:
        pass
    else:
        print("Incorrect Mode. Please Try Again")
        continue

    # Get the dimensions of the original image
    original_height, original_width, _ = img.shape

    # Get the dimensions of the resized image
    resized_height, resized_width, _ = edited_image.shape
        
    # Calculate the padding needed to match sizes
    vertical_padding = original_height - resized_height
    horizontal_padding = original_width - resized_width

    A = img.flatten()
    if mode == 4 or mode == 5:
        B = edited_image.flatten()
    #B = padding0(edited_image, vertical_padding, horizontal_padding).flatten() if vertical_padding or horizontal_padding else edited_image.flatten()
    
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1); plt.imshow(img)
    plt.title("OLD IMAGE")
    plt.subplot(2, 2, 2); plt.imshow(edited_image)
    plt.title("NEW IMAGE")
    if mode == 4 or mode == 5:
        plt.figtext(0.5, 0.3, f'Cosine Similarity : {cosine_similarity(A, B)}', fontsize=12, ha='center', va='center', color='blue')
    plt.show()
    
    
    