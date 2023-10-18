from all_import import *

def gaussian_kernel(size, sigma=1):
    if size % 2 != 1:
        return None
    size = size // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2*sigma**2))) * normal
    return g

def get_blurRGB(image:np.array):
    blur_mode = {'1':[3,0.55],'2':[3,1],'3':[5,1.5],'4':[7,2],'5':[9,3]}

    mode = input('Choose blur intensity (1-5): ')
    if not blur_mode.get(mode):
        mode = str(int(min(5, max(1, int(mode)))))
        print(f'Warning: Intensity outside of range 1-5. Using intensity of {mode}')

    ksize, ksig = blur_mode.get(mode)
    kernel = gaussian_kernel(ksize, ksig)
    
    img = image.copy()
    img = img.astype(float)

    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]

    r_blur = convolve2d(r, kernel, 'same', 'symm')
    g_blur = convolve2d(g, kernel, 'same', 'symm')
    b_blur = convolve2d(b, kernel, 'same', 'symm')

    blur = np.dstack((r_blur,g_blur,b_blur))
    blur = cv2.normalize(blur, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

    return blur
