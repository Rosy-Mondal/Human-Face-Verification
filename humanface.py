import cv2
import os
import numpy as np
from tqdm import tqdm

def preprocess_image(image_path):
    """Loads and preprocesses an image for feature extraction."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(img, (7, 7), 0)

    # Threshold the image
    _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Approximate the contour to a circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if 20 < radius < 100:  # Filter based on reasonable iris size
            return img, (int(x), int(y), int(radius))

    return None, None

def extract_features(image):
    """Extracts features using SIFT."""
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(des1, des2):
    """Matches features using FLANN-based matcher."""
    if des1 is None or des2 is None:
        return 0

    # Define FLANN parameters
    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE = 1
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test to filter good matches
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    return len(good_matches)

def find_best_match(input_image_path, folder_path):
    """Finds the best match for the input iris image in a folder."""
    input_image, input_iris = preprocess_image(input_image_path)
    if input_image is None:
        print("Input image preprocessing failed.")
        return

    _, input_descriptors = extract_features(input_image)
    max_matches = 0
    best_match_path = None

    # Iterate through folder images
    file_list = os.listdir(folder_path)
    print("Searching for the best match...")
    for filename in tqdm(file_list, desc="Matching images", ncols=80):
        file_path = os.path.join(folder_path, filename)
        candidate_image, candidate_iris = preprocess_image(file_path)
        if candidate_image is None:
            continue

        _, candidate_descriptors = extract_features(candidate_image)
        matches = match_features(input_descriptors, candidate_descriptors)

        if matches > max_matches:
            max_matches = matches
            best_match_path = file_path

    if best_match_path:
        print(f"\nBest match found: {best_match_path} with {max_matches} matches.")
        best_match_image = cv2.imread(best_match_path)
        cv2.imshow("Best Match", best_match_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("\nNo match found.")

# Example Usage
input_image_path = r'C:\Users\INDRANIL MAL\OneDrive\Desktop\New folder\python\cosamp.py\cosamp3.py\MMU-Iris-Database\1\left\aeval1.bmp'
folder_path = r'C:\Users\INDRANIL MAL\OneDrive\Desktop\New folder\python\cosamp.py\cosamp3.py\MMU-Iris-Database\1\left'
find_best_match(input_image_path, folder_path)