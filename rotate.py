from all_import import *

def rotate_image(img):
    angle = int(input("Enter rotate angle: (degrees) : "))
    h, w = img.shape[:2] # return (height, width, chanel)
    center_x, center_y = w // 2, h // 2

    # Generate Transformation Matrix
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    cos_of_rotation_matrix = np.abs(rotation_matrix[0][0])
    sin_of_rotation_matrix = np.abs(rotation_matrix[0][1])

    # Compute new size of an image
    new_w = int((h * sin_of_rotation_matrix) + (w * cos_of_rotation_matrix))
    new_h = int((h * cos_of_rotation_matrix) + (w * sin_of_rotation_matrix))
    new_center_x, new_center_y = new_w / 2, new_h / 2

    # Update Transformation Matrix New Center (x,y)
    rotation_matrix[0][2] += new_center_x - center_x
    rotation_matrix[1][2] += new_center_y - center_y

    rotated_img = cv2.warpAffine(img, rotation_matrix, (new_w, new_h))

    # Detect Image Contour and Crop an image to real size
    gray = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x_min, y_min, x_max, y_max = float('inf'), float('inf'), -float('inf'), -float('inf')

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    cropped_image = rotated_img[y_min:y_max, x_min:x_max]
    return cropped_image