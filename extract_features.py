import cv2
import numpy as np
import os
from skimage.feature import hog

def extract_features(image_path, bins=64, use_hog=False):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))
    b, g, r = cv2.split(image)

    hist_r = cv2.normalize(cv2.calcHist([r],[0],None,[bins],[0,256]), None).flatten()
    hist_g = cv2.normalize(cv2.calcHist([g],[0],None,[bins],[0,256]), None).flatten()
    hist_b = cv2.normalize(cv2.calcHist([b],[0],None,[bins],[0,256]), None).flatten()

    features = np.concatenate([hist_r, hist_g, hist_b])

    if use_hog:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hog_features = hog(gray, pixels_per_cell=(16,16), cells_per_block=(2,2))
        features = np.concatenate([features, hog_features])

    return features

# Extract features for all images
dataset_path = "dataset"
classes = ["class1", "class2"]

features = []
labels = []

for label, class_name in enumerate(classes):
    class_dir = os.path.join(dataset_path, class_name)
    image_files = [f for f in os.listdir(class_dir) if f.endswith((".jpg",".png"))]

    for img_file in image_files:
        img_path = os.path.join(class_dir, img_file)
        try:
            fv = extract_features(img_path, bins=64)
            features.append(fv)
            labels.append(label)
        except:
            continue

features = np.array(features)
labels = np.array(labels)

np.save("features.npy", features)
np.save("labels.npy", labels)
print("Features and labels saved ✅")
