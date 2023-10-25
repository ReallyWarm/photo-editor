from all_import import *
from blur import gaussian_kernel

# https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
# https://github.com/UsamaI000/CannyEdgeDetection-from-scratch-python/blob/master/CannyEdgeDetector.ipynb
# https://learnopencv.com/alpha-blending-using-opencv-cpp-python/

class EdgeOperation:
    def __init__(self, image=None):
        if image is not None:
            self.image = image.copy()
        else:
            self.image = None
        self.soft_edge = None
        self.hard_edge = None

        self.set_default()
        
    def new_image(self, image):
        self.image = image.copy()
        self.soft_edge = None
        self.hard_edge = None

    def set_default(self):
        self.gaussian_default = np.matrix([\
            [2,  4,  5,  4, 2],
            [4,  9, 12,  9, 4],
            [5, 12, 15, 12, 5],
            [4,  9, 12,  9, 4],
            [2,  4,  5,  4, 2]]) * 1/159
        
        self.new_gaussian = None
        self.low_threshold_ratio = 0.05
        self.high_threshold_ratio = 0.09
        self.weak_mag = 25
        self.strong_mag = 255

    def get_params(self):
        print('Enter Edge Detection parameters')
        mode = input('Default mode (y/n):')
        if mode.lower() in ['n', 'no']:
            g_size = int(input('Guassian kernel size (3,5,7,9,...): '))
            if g_size % 2 != 1 or g_size < 3:
                g_size = max(g_size+1, 3)
                print(f'Warning: Kernel size is not an odd number. Using size of {g_size}')
            g_sigma = float(input('Guassian kernel intensity (default=1.6): '))
            
            htr = float(input('Edge intensity high threshold ratio (default=9): '))/100
            ltr = float(input('Edge intensity low threshold ratio (default=5): '))/100

            smg = int(input('Hard edge opacity (range=0-255, default=255): '))
            if smg < 0: smg = 0
            elif smg > 255: smg = 255

            self.new_gaussian = gaussian_kernel(g_size, g_sigma)
            self.low_threshold_ratio = ltr
            self.high_threshold_ratio = htr
            self.strong_mag = smg

        elif mode.lower() in ['y', 'yes']:
            self.set_default()
        else:
            raise Exception('Incorrect Input. Please try again.')
     
    def get_edge(self):
        '''return (hard_edge, soft_edge)'''

        grey = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)

        gaussian_filter = self.new_gaussian if self.new_gaussian is not None else self.gaussian_default
        blur = convolve2d(grey, gaussian_filter, 'same', 'symm')

        grad, theta = sobel_filter(blur)
        grad = cv2.convertScaleAbs(grad)

        angle = gradient_direction(theta)
        nms = non_max_suppression(grad, angle)
        self.soft_edge = nms.copy()

        thresh, weak, strong = double_thresholding(nms, self.low_threshold_ratio, self.high_threshold_ratio, self.weak_mag, self.strong_mag)
        canny = hysteresis(thresh, weak, strong)
        self.hard_edge = canny.copy()

        return self.hard_edge, self.soft_edge
    
    def edge_enhance(self):
        hard, soft = self.get_edge()

        print('Choose Enhance Mode')
        print('[1]: Soft edge')
        print('[2]: Hard edge')
        mode = int(input('Please Select Mode: '))
        if mode not in [1,2]:
            raise Exception('Incorrect Mode. Please Try Again')
        
        color = [max(0, min(int(x), 255)) for x in input("Input color of edges (RGB) ex. xxx xxx xxx : ").split(' ')]
        edge = soft if mode == 1 else hard

        # Alpha Blending Method
        edge_rgb = cv2.cvtColor(edge.copy(), cv2.COLOR_GRAY2RGB)
        # Normalize the alpha mask to keep intensity between 0 and 1
        alpha = edge_rgb.astype(float) / 255

        color_img = np.zeros(self.image.shape)
        color_img[:] = color
        foreground = color_img.astype(float)

        background = self.image.astype(float)
        
        foreground = cv2.multiply(alpha, foreground)
        background = cv2.multiply(1.0-alpha, background)
        out = cv2.add(foreground, background) / 255

        out = cv2.normalize(out, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

        return out


def sobel_filter(grey_img):
    sobelx = np.matrix([[-1, 0, 1], 
                        [-2, 0, 2], 
                        [-1, 0, 1]], np.float32)
    
    sobely = np.matrix([[ 1, 2, 1], 
                        [ 0, 0, 0], 
                        [-1,-2,-1]], np.float32)
    
    Gx = convolve2d(grey_img, sobelx, "same", "symm")
    Gy = convolve2d(grey_img, sobely, "same", "symm")

    Gout = np.hypot(Gx, Gy) #sqrt(Gx**2 + Gy**2)
    theta = np.arctan2(Gy, Gx)

    return Gout, theta

def gradient_direction(theta):
    angle = np.rad2deg(theta) + 180
    return angle

def non_max_suppression(grad, angle):
    h, w = grad.shape
    nms = np.zeros((h, w), dtype=np.uint8)
    
    for i in range(1, h-1):
        for j in range(1, w-1):
            q = 255
            r = 255
            
            # angle 0 and 180
            if (0 <= angle[i,j] < 22.5) or (337.5 <= angle[i,j] <= 360) or (157.5 <= angle[i,j] < 202.5):
                q = grad[i,j+1]
                r = grad[i,j-1]
            # angle 45 and 225
            elif (22.5 <= angle[i,j] < 67.5) or (202.5 <= angle[i, j] < 247.5):
                q = grad[i+1,j-1]
                r = grad[i-1,j+1]
            # angle 90 and 270
            elif (67.5 <= angle[i,j] < 112.5) or (247.5 <= angle[i, j] < 292.5):
                q = grad[i+1,j]
                r = grad[i-1,j]
            # angle 135 and 315
            elif (112.5 <= angle[i,j] < 157.5) or (292.5 <= angle[i, j] < 337.5):
                q = grad[i-1,j-1]
                r = grad[i+1,j+1]

            if (grad[i,j] >= q) and (grad[i,j] >= r):
                nms[i,j] = grad[i,j]
            else:
                nms[i,j] = 0
    
    return nms

def double_thresholding(suppressed, low_ratio=0.05, high_ratio=0.09, weak=25, strong=255):
    high_threshold = suppressed.max() * high_ratio
    low_threshold = high_threshold * low_ratio

    h, w = suppressed.shape
    thresholded = np.zeros((h, w), dtype=np.uint8)

    strong_i, strong_j = np.where(suppressed >= high_threshold)
    weak_i, weak_j = np.where((suppressed <= high_threshold) & (suppressed >= low_threshold))

    thresholded[strong_i, strong_j] = strong
    thresholded[weak_i, weak_j] = weak

    return thresholded, weak, strong

def hysteresis(threshold, weak=25, strong=255):
    h, w = threshold.shape
    strong_edge = np.zeros((h, w), dtype=np.uint8)

    for i in range(1, h-1):
        for j in range(1, w-1):

            if threshold[i,j] == weak:
                if (threshold[i-1,j] == strong) or (threshold[i-1,j+1] == strong) or (threshold[i,j+1] == strong) or (threshold[i+1,j+1] == strong) or \
                   (threshold[i+1,j] == strong) or (threshold[i+1,j-1] == strong) or (threshold[i,j-1] == strong) or (threshold[i-1,j-1] == strong):
                    strong_edge[i,j] = strong

            elif threshold[i,j] == strong:
                strong_edge[i,j] = strong

    return strong_edge
