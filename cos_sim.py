from all_import import *
from sklearn.metrics.pairwise import cosine_similarity

def padding0(edited_image, vert_padding, hori_padding):
        # Pad the resized image with zeros
        resized_padded = cv2.copyMakeBorder(edited_image, vert_padding // 2, vert_padding // 2, hori_padding // 2, hori_padding // 2, cv2.BORDER_CONSTANT, None, value=0)
        return resized_padded

def cosine_similarity(vector1, vector2):
    cos_sim = 1 - spatial.distance.cosine(vector1, vector2)
    return cos_sim