from all_import import spatial

def cosine_similarity(vector1, vector2):
    cos_sim = 1 - spatial.distance.cosine(vector1, vector2)
    return cos_sim