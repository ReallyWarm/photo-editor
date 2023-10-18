from edge_detect import EdgeOperation, gaussian_kernel
from color_threshold import color_threshold
from rotate import rotate_image
from resize_crop import resize_image, croping
from cos_sim import cosine_similarity, padding0
from all_import import *

def display(img, name='win'):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

mode_list = ["Resize", "Rotate", "Crop", "Color Threshold", "Edge Enhance"]
edge_op = EdgeOperation()

if __name__ == '__main__':
    img_name = input("Enter Image Name (Ex. image.jpg): ")
    try:
        img = cv2.imread(f"imgin/{img_name}")
    except cv2.error as e:
        pass

    if img is None: 
        raise Exception(f'Can\'t open/read file from (.imgin/{img_name}): Please check file path/integrity.')

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    edge_op.new_image(img)
    
    print("Choose Features to Edit Image")
    for i in range(len(mode_list)):
        print(f"[{i + 1}]: {mode_list[i]}")

    mode = int(input("Please Select Feature : "))
    if mode == 1:
        edited_image = resize_image(img)
    elif mode == 2:
        edited_image = rotate_image(img)
    elif mode == 3:
        edited_image = croping(img)
    elif mode == 4:
        edited_image = color_threshold(img)
    elif mode == 5:
        edge_op.get_params()
        edited_image = edge_op.edge_enhance()
    else:
        raise Exception("Incorrect Mode: Please Try Again")

    # Get the dimensions of the original image
    original_height, original_width, _ = img.shape

    # Get the dimensions of the resized image
    resized_height, resized_width, _ = edited_image.shape
        
    # Calculate the padding needed to match sizes
    vertical_padding = abs(original_height - resized_height)
    horizontal_padding = abs(original_width - resized_width)

    # Padding a smaller image to match image size
    if mode == 4 or mode == 5:
        A = img.flatten()
        B = edited_image.flatten()
    else:
        if (original_height > resized_height) or (original_width > resized_width):
            padding_image = padding0(edited_image, vertical_padding, horizontal_padding)
            A = img.flatten()
            B = padding_image.flatten()
        else:
            padding_image = padding0(img, vertical_padding, horizontal_padding)
            A = padding_image.flatten()
            B = edited_image.flatten()

    # Display images
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1); plt.imshow(img)
    plt.title("OLD IMAGE")
    plt.subplot(2, 2, 2); plt.imshow(edited_image)
    plt.title("NEW IMAGE")
    plt.figtext(0.5, 0.3, f'Cosine Similarity : {cosine_similarity(A, B)}', fontsize=12, ha='center', va='center', color='blue')
    plt.show()