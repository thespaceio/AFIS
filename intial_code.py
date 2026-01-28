# This code should still be kept safe as a prototype for what was made

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to load the image
def load_fingerprint(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not find image to display: {image_path}")
    return img

#Sample or Test

img = load_fingerprint (r"C:\Users\USER\Desktop\AFIS\SOCOFing\Real\64__M_Right_index_finger.BMP")
plt.imshow(img, cmap='gray')
plt.title("Original Fingerprint")
plt.show()


# Function to Enhance Preprocessing Pipeline

def enhance_fingerprint(image):

    #normalizing the intensity
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    # Applying CLAHE for local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized)

    #Removing noise with non-local means donoising
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)

    # Applying Gabor filter bank to enhance ridge structure
    gabor_kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    filtered = cv2.filter2D(denoised, cv2.CV_8UC3, gabor_kernel)

    # Adaptive Threshold
    binary = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    return cleaned, filtered


# To test the pipeline
img = load_fingerprint(r"C:\Users\USER\Desktop\AFIS\SOCOFing\Real\64__M_Right_index_finger.BMP")
binary_img, enhanced_img = enhance_fingerprint(img)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img, cmap='gray')
axes[0].set_title("Original")
axes[1].imshow(enhanced_img, cmap='gray')
axes[1].set_title("Enhanced")
axes[2].imshow(binary_img, cmap='gray')
axes[2].set_title("Binary")
plt.show()


def skeletonize_image(binary_image):

    from skimage.morphology import skeletonize
    # Invert for skeletonization (needs white ridges on black background)
    inverted = cv2.bitwise_not(binary_image)
    # Convert to boolean array
    binary_bool = inverted > 0
    # Apply skeletonization
    skeleton = skeletonize(binary_bool)
    # Convert back to unit8
    skeleton_unit8 = (skeleton.astype(np.unit8)) * 255

    return skeleton_unit8

skeleton = skeletonize_image(binary_img)
plt.imshow(skeleton, cmap='gray')
plt.title("Skeletonized Ridges")
plt.show()


def extract_minutiae(skeleton_img):

    minutiae_points = []
    rows, cols = skeleton_img.shape

    # Defining 8 connectivity neighbors
    neighbors = [(0, 1), (1, 1), (1, 0), (1, -1),
                 (0, -1), (-1, -1), (-1, 0), (-1, 1)]

    #pad the image to handle borders
    padded = cv2.copyMakeBorder(skeleton_img, 1,1,1,1, cv2.BORDER_CONSTANT, value=0)

    for y in range(1, rows+1):
        for x in range(1, cols+1):
            if padded[y, x] == 255:  #Ridge Pixel
                #Count neighbouring ridge pixels
                ridge_neighbors = 0
                for dx, dy in neighbors:
                    if padded[y + dy, x + dx] == 255:
                        ridge_neighbors += 1

                # Minutiae Classfication
                if ridge_neighbors == 1:        # RIdge ending
                    minutiae_points.append({
                        'x': x-1, 'y': y-1,
                        'type': 'ending',
                        'angle': calculate_orientation(padded, x, y)
                    })

                elif ridge_neighbors == 3:          #Bifurcation
                    minutiae_points.append({
                        'x': x-1, 'y': y-1,
                        'type': 'bifurcation',
                        'angle': calculate_orientation(padded, x, y)
                    })

    return minutiae_points


def calculate_orientation(img, x, y):



    try:

        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        orientation = np.arctan(sobel_y[y, x], sobel_x[y, x])
        return orientation
    except:
        return 0.0


# Extracting minutiae

minutiae = extract_minutiae(skeleton)
print(f"Found {len(minutiae)} minutiae points")

# visualize minutiae

def visualize_minutiae(original_img, minutiae_points):
    """Draw minutiae on the original image"""
    img_color = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    
    for point in minutiae_points:
        x, y = int(point['x']), int(point['y'])
        color = (0, 255, 0) if point['type'] == 'ending' else (255, 0, 0)  # Green for endings, Blue for bifurcations
        cv2.circle(img_color, (x, y), 3, color, -1)
        
        # Draw orientation line
        angle = point['angle']
        length = 10
        end_x = int(x + length * np.cos(angle))
        end_y = int(y + length * np.sin(angle))
        cv2.line(img_color, (x, y), (end_x, end_y), color, 1)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    plt.title("Minutiae Points (Green: Endings, Blue: Bifurcations)")
    plt.show()

visualize_minutiae(img, minutiae[:50])  # Show first 50 minutiae

