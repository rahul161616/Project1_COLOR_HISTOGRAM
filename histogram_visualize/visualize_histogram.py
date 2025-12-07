import cv2
import matplotlib.pyplot as plt
import os

# Correct folder path (replace with your actual folder)
class1_dir = "../dataset/class2"  

# List all files in folder
image_files = os.listdir(class1_dir)

# Pick the first image (ensure it’s a valid image file)
image_path = os.path.join(class1_dir, image_files[0])
print("Loading image:", image_path)

# Load image
image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"Cannot load image: {image_path}")

# Resize
image = cv2.resize(image, (256, 256))

# Split channels
b, g, r = cv2.split(image)

# Plot histograms
colors = ('b', 'g', 'r')
channels = [b, g, r]

plt.figure(figsize=(10,5))
for i, col in enumerate(colors):
    hist = cv2.calcHist([channels[i]], [0], None, [256], [0,256])
    plt.plot(hist, color=col)
    plt.xlim([0,256])

plt.title("RGB Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.show()

