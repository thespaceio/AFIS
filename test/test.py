import cv2
import numpy as np


def test_image_loading(image_path):
    """Test if your image can be loaded and enhanced"""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    print(f"Original: {image.shape[1]}x{image.shape[0]}, contrast: {np.std(image):.2f}")

    # Test resizing
    if image.shape[0] < 200 or image.shape[1] < 200:
        scale = max(200 / image.shape[0], 200 / image.shape[1])
        new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
        resized = cv2.resize(image, new_size, cv2.INTER_CUBIC)
        print(f"After resize: {resized.shape[1]}x{resized.shape[0]}, contrast: {np.std(resized):.2f}")
        image = resized

    # Test contrast enhancement
    if np.std(image) < 20:
        print("Applying CLAHE...")
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        print(f"After CLAHE: contrast: {np.std(enhanced):.2f}")

        if np.std(enhanced) < 20:
            print("Applying histogram equalization...")
            equalized = cv2.equalizeHist(enhanced)
            print(f"After equalization: contrast: {np.std(equalized):.2f}")
            image = equalized
        else:
            image = enhanced

    print(f"Final contrast: {np.std(image):.2f}")
    print(f"PASS" if np.std(image) >= 15 else "FAIL")
    return image


# Test your image
test_image_loading(r"C:\Users\USER\Desktop\AFIS\SOCOFing\Real\1__M_Left_index_finger.BMP")
