from all_import import *

def padding0(edited_image, vert_padding, hori_padding):
        # Pad the resized image with zeros
        if vert_padding % 2 == 1 and hori_padding % 2 == 1:               # vertical ODD, Horizon is ODD
              resized_padded = cv2.copyMakeBorder(edited_image, vert_padding // 2 + 1, vert_padding // 2, hori_padding // 2 + 1, hori_padding // 2, cv2.BORDER_CONSTANT, None, value=0)
        elif vert_padding % 2 == 1 and hori_padding % 2 == 0:               # vertical ODD , Horizon Even
                resized_padded = cv2.copyMakeBorder(edited_image, vert_padding // 2 + 1, vert_padding // 2, hori_padding // 2, hori_padding // 2, cv2.BORDER_CONSTANT, None, value=0)
        elif vert_padding % 2 == 0 and hori_padding % 2 == 1:               # vertical EVEN , Horizon is ODD
                resized_padded = cv2.copyMakeBorder(edited_image, vert_padding // 2, vert_padding // 2, hori_padding // 2 + 1, hori_padding // 2, cv2.BORDER_CONSTANT, None, value=0)
        else:                                                               # vertiacl EVEN , Horizon EVEN
              resized_padded = cv2.copyMakeBorder(edited_image, vert_padding // 2, vert_padding // 2, hori_padding // 2, hori_padding // 2, cv2.BORDER_CONSTANT, None, value=0)

        return resized_padded

def cosine_similarity(vector1, vector2):
    # Cosine Distance = 1 - ((u dot v) / |u|^2 |v|^2)
    cos_sim = 1 - spatial.distance.cosine(vector1, vector2) # Cosine Similarity = ((u dot v) / |u|^2 |v|^2) = 1 - Cosine Distance
    return cos_sim