# import cv2
# import numpy as np
# import joblib
# import os
# import sys


# model_path = "../knn_color_hist_model.pkl"
# if not os.path.exists(model_path):
#     sys.exit(f"Error: model file not found at {model_path}")


# def extract_color_histogram(image_path, bins=32):
#     image = cv2.imread(image_path)
#     image = cv2.resize(image, (256, 256))
#     b, g, r = cv2.split(image)
#     hist_r = cv2.normalize(cv2.calcHist([r],[0],None,[bins],[0,256]), None).flatten()
#     hist_g = cv2.normalize(cv2.calcHist([g],[0],None,[bins],[0,256]), None).flatten()
#     hist_b = cv2.normalize(cv2.calcHist([b],[0],None,[bins],[0,256]), None).flatten()
#     return np.concatenate([hist_r, hist_g, hist_b])

# # Load trained model
# knn = joblib.load("../knn_color_hist_model.pkl")

# # Predict
# image_path = "art-moonlight-tropical-sea-beach-night-vacation-palms-resort-78380984.webp"  # replace with your test image
# features = extract_color_histogram(image_path).reshape(1, -1)
# pred = knn.predict(features)
# print("Predicted class:", pred[0])
import os, cv2, numpy as np, joblib

def extract_features(image_path, bins=64):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))
    b, g, r = cv2.split(image)
    hist_r = cv2.normalize(cv2.calcHist([r],[0],None,[bins],[0,256]), None).flatten()
    hist_g = cv2.normalize(cv2.calcHist([g],[0],None,[bins],[0,256]), None).flatten()
    hist_b = cv2.normalize(cv2.calcHist([b],[0],None,[bins],[0,256]), None).flatten()
    return np.concatenate([hist_r,hist_g,hist_b])

model_path = os.path.join(os.path.dirname(__file__), "..", "knn_color_hist_model.pkl")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")
knn = joblib.load(model_path)

provided_image = os.path.join(os.path.dirname(__file__), "art-moonlight-tropical-sea-beach-night-vacation-palms-resort-78380984.webp")
if not os.path.exists(provided_image):
    raise FileNotFoundError(f"Image not found: {provided_image}")

feat = extract_features(provided_image).reshape(1,-1)
pred = knn.predict(feat)[0]

img = cv2.imread(provided_image)
img = cv2.cvtColor(cv2.resize(img,(256,256)), cv2.COLOR_BGR2RGB)

import matplotlib.pyplot as plt
plt.imshow(img)
plt.axis('off')
plt.title(f"Prediction: {['class1','class2'][pred]}")
plt.show()
