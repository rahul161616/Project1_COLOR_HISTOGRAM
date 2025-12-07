# import os
# import cv2
# import numpy as np
# import joblib
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# # Feature extraction function
# def extract_color_histogram(image_path, bins=32):
#     image = cv2.imread(image_path)
#     image = cv2.resize(image, (256, 256))
#     b, g, r = cv2.split(image)
#     hist_r = cv2.normalize(cv2.calcHist([r],[0],None,[bins],[0,256]), None).flatten()
#     hist_g = cv2.normalize(cv2.calcHist([g],[0],None,[bins],[0,256]), None).flatten()
#     hist_b = cv2.normalize(cv2.calcHist([b],[0],None,[bins],[0,256]), None).flatten()
#     return np.concatenate([hist_r, hist_g, hist_b])

# # Load trained model
# knn = joblib.load("../knn_color_hist_model.pkl")  # adjust path if needed

# # Dataset path
# dataset_path = "../dataset"
# classes = ["class1", "class2"]

# y_true = []
# y_pred = []

# # Store images for visualization
# images_class1 = []
# images_class2 = []

# # Loop through all images and predict
# for label, class_name in enumerate(classes):
#     class_dir = os.path.join(dataset_path, class_name)
#     image_files = [f for f in os.listdir(class_dir) if f.endswith((".jpg", ".png", ".webp"))]

#     for img_file in image_files:
#         img_path = os.path.join(class_dir, img_file)
#         features = extract_color_histogram(img_path).reshape(1, -1)
#         pred = knn.predict(features)[0]

#         y_true.append(label)
#         y_pred.append(pred)

#         # Save image for plotting
#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(cv2.resize(img, (64, 64)), cv2.COLOR_BGR2RGB)
#         if label == 0:
#             images_class1.append((img, pred))
#         else:
#             images_class2.append((img, pred))

# # Convert to numpy arrays
# y_true = np.array(y_true)
# y_pred = np.array(y_pred)

# # Accuracy and classification report
# accuracy = accuracy_score(y_true, y_pred)
# print(f"Overall Accuracy: {accuracy*100:.2f}%\n")
# print("Classification Report:")
# print(classification_report(y_true, y_pred, target_names=classes))

# # Confusion Matrix
# cm = confusion_matrix(y_true, y_pred)
# print("Confusion Matrix:")
# print(cm)

# # Plot confusion matrix
# plt.figure(figsize=(5,5))
# plt.imshow(cm, cmap=plt.cm.Blues)
# plt.colorbar()
# plt.xticks([0,1], classes)
# plt.yticks([0,1], classes)
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix")
# plt.show()

# # Function to plot sample images with predictions
# def plot_sample_images(images, class_name):
#     plt.figure(figsize=(12,4))
#     for i, (img, pred) in enumerate(images[:8]):  # plot first 8 images
#         plt.subplot(2,4,i+1)
#         plt.imshow(img)
#         plt.axis('off')
#         plt.title(f"Pred: {classes[pred]}")
#     plt.suptitle(f"Sample images from {class_name}")
#     plt.show()

# plot_sample_images(images_class1, classes[0])
# plot_sample_images(images_class2, classes[1])

# # Predict a single provided image
# provided_image = "art-moonlight-tropical-sea-beach-night-vacation-palms-resort-78380984.webp"  # replace with your image path
# features = extract_color_histogram(provided_image).reshape(1,-1)
# pred = knn.predict(features)[0]

# img = cv2.imread(provided_image)
# img = cv2.cvtColor(cv2.resize(img, (256,256)), cv2.COLOR_BGR2RGB)
# plt.imshow(img)
# plt.axis('off')
# plt.title(f"Prediction for Provided Image: {classes[pred]}")
# plt.show(block=False)
# plt.pause(0.1)  # small pause to render the figure
# plt.close()     # close the figure programmatically

import os, cv2, numpy as np, joblib, matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Feature extraction
def extract_features(image_path, bins=64):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))
    b, g, r = cv2.split(image)
    hist_r = cv2.normalize(cv2.calcHist([r],[0],None,[bins],[0,256]), None).flatten()
    hist_g = cv2.normalize(cv2.calcHist([g],[0],None,[bins],[0,256]), None).flatten()
    hist_b = cv2.normalize(cv2.calcHist([b],[0],None,[bins],[0,256]), None).flatten()
    return np.concatenate([hist_r, hist_g, hist_b])

# Load model
model_path = os.path.join(os.path.dirname(__file__), "..", "knn_color_hist_model.pkl")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")
knn = joblib.load(model_path)

dataset_path = os.path.join(os.path.dirname(__file__), "..", "dataset")
classes = ["class1","class2"]
y_true, y_pred = [], []
images_class1, images_class2 = [], []

for label, cls in enumerate(classes):
    folder = os.path.join(dataset_path, cls)
    image_files = [f for f in os.listdir(folder) if f.endswith((".jpg",".png"))]
    for f in image_files:
        path = os.path.join(folder,f)
        feat = extract_features(path).reshape(1,-1)
        pred = knn.predict(feat)[0]
        y_true.append(label)
        y_pred.append(pred)
        img = cv2.imread(path)
        img = cv2.cvtColor(cv2.resize(img,(64,64)), cv2.COLOR_BGR2RGB)
        if label==0: images_class1.append((img,pred))
        else: images_class2.append((img,pred))

print("\n" + "="*60)
print("ANALYZING THE PROVIDED INPUT IMAGE")
print("="*60 + "\n")

# Use the same image as predict.py
input_image_path = os.path.join(os.path.dirname(__file__), "art-moonlight-tropical-sea-beach-night-vacation-palms-resort-78380984.webp")

if os.path.exists(input_image_path):
    feat = extract_features(input_image_path).reshape(1,-1)
    pred = knn.predict(feat)[0]
    
    print(f"📷 Image: {os.path.basename(input_image_path)}")
    print(f"🎯 Prediction: {classes[pred]}")
    print(f"📊 Class 0 (Beach) | Class 1 (Forest)")
    
    # Get confidence scores
    distances, indices = knn.kneighbors(feat)
    print(f"   KNN neighbors found at distances: {distances[0]}")
    print(f"   Neighbor classes: {knn._fit_X[indices[0]]}")
    
    # Display image
    img = cv2.imread(input_image_path)
    img = cv2.cvtColor(cv2.resize(img,(256,256)), cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(8,6))
    plt.imshow(img)
    plt.axis('off')
    pred_label = classes[pred]
    color = 'green' if pred == 0 else 'blue'
    plt.title(f"Input Image Prediction: {pred_label}", fontsize=14, color=color, fontweight='bold')
    print("\n[Close the plot to continue...]")
    plt.show()
else:
    print(f"⚠️ Input image not found at {input_image_path}")

print("\n" + "="*60)
print("FULL DATASET EVALUATION")
print("="*60 + "\n")

# Add text annotations to confusion matrix
for i in range(2):
    for j in range(2):
        text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
        plt.text(j, i, cm[i, j], ha='center', va='center', color=text_color, fontsize=20)

# Plot confusion matrix
plt.figure(figsize=(8,6))
plt.imshow(cm, cmap=plt.cm.Blues)
plt.colorbar()
plt.xticks([0,1], classes)
plt.yticks([0,1], classes)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.title("Confusion Matrix", fontsize=14)
for i in range(2):
    for j in range(2):
        text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
        plt.text(j, i, cm[i, j], ha='center', va='center', color=text_color, fontsize=20)
print("\n[Press any key on the plot window to continue...]")
plt.show()

# Plot sample images per class
def plot_samples(images, cls_name):
    plt.figure(figsize=(12,4))
    for i,(img,pred) in enumerate(images[:8]):
        plt.subplot(2,4,i+1)
        plt.imshow(img)
        plt.axis('off')
        pred_label = classes[pred]
        color = 'green' if pred == (0 if cls_name == classes[0] else 1) else 'red'
        plt.title(f"Pred: {pred_label}", color=color, fontsize=10)
    plt.suptitle(f"Samples from {cls_name}", fontsize=14)
    print(f"\n[Showing samples from {cls_name} - Press any key on plot to continue...]")
    plt.show()

plot_samples(images_class1, classes[0])
plot_samples(images_class2, classes[1])

print("\n✅ Testing complete! All visualizations shown.")
