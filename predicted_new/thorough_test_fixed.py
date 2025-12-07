import os
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Feature extraction with 64 bins to match training
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
classes = ["class1 (Beach)", "class2 (Forest)"]

print("\n" + "="*70)
print("ANALYZING THE PROVIDED INPUT IMAGE WITH predict.py")
print("="*70 + "\n")

# Analyze the provided input image
input_image_path = os.path.join(os.path.dirname(__file__), "art-moonlight-tropical-sea-beach-night-vacation-palms-resort-78380984.webp")

if os.path.exists(input_image_path):
    feat = extract_features(input_image_path).reshape(1, -1)
    pred = knn.predict(feat)[0]
    
    print(f"📷 Input Image: art-moonlight-tropical-sea-beach-night-vacation-palms-resort-78380984.webp")
    print(f"🎯 Model Prediction: {classes[pred]}")
    print(f"✨ Confidence: The model classified this as {classes[pred]}")
    
    # Display the input image with prediction
    img = cv2.imread(input_image_path)
    img = cv2.cvtColor(cv2.resize(img, (256, 256)), cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis('off')
    pred_label = classes[pred]
    color = 'darkgreen' if pred == 0 else 'darkblue'
    plt.title(f"Input Image Analysis: Predicted as {pred_label}", fontsize=14, color=color, fontweight='bold')
    print("\n[Close the plot window to continue to full dataset evaluation...]")
    plt.show()
else:
    print(f"⚠️ Input image not found at {input_image_path}")

print("\n" + "="*70)
print("FULL DATASET EVALUATION (All 120 Images)")
print("="*70 + "\n")

# Evaluate on full dataset
y_true = []
y_pred = []
images_class1 = []
images_class2 = []

for label, cls in enumerate(["class1", "class2"]):
    folder = os.path.join(dataset_path, cls)
    image_files = [f for f in os.listdir(folder) if f.endswith((".jpg", ".png"))]
    for f in image_files:
        path = os.path.join(folder, f)
        feat = extract_features(path).reshape(1, -1)
        pred = knn.predict(feat)[0]
        y_true.append(label)
        y_pred.append(pred)
        img = cv2.imread(path)
        img = cv2.cvtColor(cv2.resize(img, (64, 64)), cv2.COLOR_BGR2RGB)
        if label == 0:
            images_class1.append((img, pred))
        else:
            images_class2.append((img, pred))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Print accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Overall Accuracy: {accuracy*100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=["class1 (Beach)", "class2 (Forest)"]))

cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap=plt.cm.Blues)
plt.colorbar()
plt.xticks([0, 1], ["Beach (0)", "Forest (1)"])
plt.yticks([0, 1], ["Beach (0)", "Forest (1)"])
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.title("Confusion Matrix - Full Dataset Evaluation", fontsize=14, fontweight='bold')
for i in range(2):
    for j in range(2):
        text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
        plt.text(j, i, cm[i, j], ha='center', va='center', color=text_color, fontsize=20, fontweight='bold')
print("[Close the plot to continue...]")
plt.show()

# Plot sample images per class
def plot_samples(images, cls_name, cls_idx):
    plt.figure(figsize=(14, 5))
    for i, (img, pred) in enumerate(images[:8]):
        plt.subplot(2, 4, i + 1)
        plt.imshow(img)
        plt.axis('off')
        is_correct = (pred == cls_idx)
        color = 'darkgreen' if is_correct else 'darkred'
        status = "✓" if is_correct else "✗"
        plt.title(f"Pred: {classes[pred]}\n{status}", color=color, fontsize=9, fontweight='bold')
    plt.suptitle(f"Sample Predictions from {cls_name}", fontsize=14, fontweight='bold')
    print(f"[Close the plot to continue...]")
    plt.show()

plot_samples(images_class1, "Class1 (Beach)", 0)
plot_samples(images_class2, "Class2 (Forest)", 1)

print("\n" + "="*70)
print("✅ TESTING COMPLETE!")
print("="*70)
