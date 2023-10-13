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

    fig, ax = plt.subplots(1,2,figsize=(12,6))
    ax[0].imshow(img)
    ax[1].imshow(edited_image)
    plt.show()
    
    
    