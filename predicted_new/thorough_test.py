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

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("Accuracy:", accuracy_score(y_true,y_pred))
print("Classification Report:\n", classification_report(y_true,y_pred,target_names=classes))
cm = confusion_matrix(y_true,y_pred)

# Plot confusion matrix
plt.figure(figsize=(5,5))
plt.imshow(cm, cmap=plt.cm.Blues)
plt.colorbar()
plt.xticks([0,1], classes)
plt.yticks([0,1], classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
for i in range(2):
    for j in range(2):
        text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
        plt.text(j, i, cm[i, j], ha='center', va='center', color=text_color, fontsize=20)
print("\n[Close the confusion matrix plot to continue...]")
plt.show()

# Plot sample images per class
def plot_samples(images, cls_name):
    plt.figure(figsize=(12,4))
    for i,(img,pred) in enumerate(images[:8]):
        plt.subplot(2,4,i+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Pred:{classes[pred]}")
    plt.suptitle(f"Samples: {cls_name}")
    print(f"[Close the plot to continue...]")
    plt.show()

plot_samples(images_class1, classes[0])
plot_samples(images_class2, classes[1])

print("\n✅ Testing complete!")
