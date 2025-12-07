# import cv2
# import numpy as np
# import os

# def extract_color_histogram(image_path, bins=32):
#     image = cv2.imread(image_path)
#     image = cv2.resize(image, (256, 256))

#     # Split channels
#     b, g, r = cv2.split(image)

#     # Compute histograms
#     hist_r = cv2.calcHist([r], [0], None, [bins], [0, 256])
#     hist_g = cv2.calcHist([g], [0], None, [bins], [0, 256])
#     hist_b = cv2.calcHist([b], [0], None, [bins], [0, 256])

#     # Normalize
#     hist_r = cv2.normalize(hist_r, hist_r).flatten()
#     hist_g = cv2.normalize(hist_g, hist_g).flatten()
#     hist_b = cv2.normalize(hist_b, hist_b).flatten()

#     # Combine
#     feature_vector = np.concatenate([hist_r, hist_g, hist_b])
#     return feature_vector

# # Paths
# dataset_path = "dataset"
# classes = ["class1", "class2"]

# features = []
# labels = []

# for label, class_name in enumerate(classes):
#     class_dir = os.path.join(dataset_path, class_name)
#     image_files = [f for f in os.listdir(class_dir) if f.endswith((".jpg", ".png"))]

#     for img_file in image_files:
#         img_path = os.path.join(class_dir, img_file)
#         try:
#             fv = extract_color_histogram(img_path)
#             features.append(fv)
#             labels.append(label)
#         except Exception as e:
#             print(f"Error processing {img_path}: {e}")

# # Convert to numpy arrays
# features = np.array(features)
# labels = np.array(labels)

# print("Feature matrix shape:", features.shape)
# print("Labels shape:", labels.shape)

# # Save features and labels for training
# np.save("features.npy", features)
# np.save("labels.npy", labels)

# print("Features and labels saved successfully ✅")
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
