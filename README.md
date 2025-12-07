# 🌊🌲 Color Histogram Image Classification

A machine learning project that classifies landscape images into **Beach** and **Forest** categories using color histogram features and K-Nearest Neighbors (KNN) algorithm.

## 📊 Project Overview

This project demonstrates image classification using computer vision and machine learning techniques. It extracts RGB color histogram features from images and uses a KNN classifier to distinguish between beach and forest landscapes.

### 🎯 Key Features

- **Automated Image Collection**: Download images from Bing using web scraping
- **Feature Extraction**: RGB color histogram analysis (64 bins per channel = 192 features)
- **Machine Learning**: K-Nearest Neighbors classifier
- **Model Evaluation**: Comprehensive testing with confusion matrix and visualizations
- **Prediction**: Single image classification with visual output

## 🏆 Performance Metrics

| Metric | Class1 (Beach) | Class2 (Forest) | Overall |
|--------|---------------|-----------------|---------|
| **Precision** | 89% | 86% | 87% |
| **Recall** | 78% | 93% | 87% |
| **F1-Score** | 83% | 89% | 87% |
| **Accuracy** | - | - | **86.78%** |

**Dataset**: 121 images (51 beach + 70 forest)

## 🗂️ Project Structure

```
project1_color_histogram/
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore file
│
├── download_images.py              # Download images from Bing
├── extract_features.py             # Extract color histogram features
├── train_model.py                  # Train KNN classifier
├── test_env.py                     # Test environment setup
│
├── dataset/                        # Image dataset (not in repo)
│   ├── class1/                     # Beach landscape images
│   └── class2/                     # Forest landscape images
│
├── histogram_visualize/
│   └── visualize_histogram.py      # Visualize RGB histograms
│
├── predicted_new/
│   ├── predict.py                  # Single image prediction
│   ├── thorough_test_fixed.py      # Comprehensive model evaluation
│   └── *.webp                      # Sample test images
│
├── features.npy                    # Extracted features (not in repo)
├── labels.npy                      # Image labels (not in repo)
└── knn_color_hist_model.pkl        # Trained model (not in repo)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.13+ (tested on 3.13.9)
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd project1_color_histogram
```

2. **Create and activate virtual environment**
```bash
python3 -m venv histo
source histo/bin/activate  # On Linux/Mac
# histo\Scripts\activate   # On Windows
```

3. **Install dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Dependencies

```
opencv-python==4.12.0.88        # Image processing
scikit-learn==1.7.2             # Machine learning
scikit-image==0.25.2            # Advanced image features
numpy==2.2.6                    # Numerical operations
matplotlib==3.10.7              # Visualization
scipy==1.16.3                   # Scientific computing
pillow==12.0.0                  # Image handling
icrawler==0.6.10                # Image downloading
beautifulsoup4==4.14.3          # Web scraping
joblib==1.5.2                   # Model serialization
```

## 📖 Usage Guide

### 1. Download Images (Optional)

Download beach and forest images from Bing:

```bash
python download_images.py
```

This will download:
- ~30 beach landscape images → `dataset/class1/`
- ~30 forest landscape images → `dataset/class2/`

### 2. Extract Features

Extract RGB color histogram features from images:

```bash
python extract_features.py
```

**Output**:
- `features.npy` - Feature matrix (N × 192)
- `labels.npy` - Image labels (N,)

### 3. Train Model

Train the KNN classifier:

```bash
python train_model.py
```

**Output**:
- `knn_color_hist_model.pkl` - Trained model
- Training accuracy and classification report

### 4. Test Environment

Verify installation:

```bash
python test_env.py
```

### 5. Predict Single Image

Classify a single image:

```bash
python predicted_new/predict.py
```

Displays the image with prediction overlay.

### 6. Comprehensive Testing

Run full evaluation with visualizations:

```bash
python predicted_new/thorough_test_fixed.py
```

**Shows**:
1. Input image prediction
2. Overall accuracy metrics
3. Confusion matrix
4. Sample predictions from both classes

### 7. Visualize Histograms

View RGB color distribution:

```bash
python histogram_visualize/visualize_histogram.py
```

## 🔬 Technical Details

### Feature Extraction Algorithm

1. **Image Preprocessing**
   - Resize to 256×256 pixels
   - Split into R, G, B channels

2. **Histogram Computation**
   - Compute 64-bin histogram per channel
   - Intensity range: 0-256

3. **Normalization**
   - Normalize each histogram independently
   - Flatten to 1D arrays

4. **Concatenation**
   - Combine R+G+B histograms
   - Final feature vector: 192 dimensions

### Model Architecture

- **Algorithm**: K-Nearest Neighbors (KNN)
- **Parameters**: k=3 neighbors
- **Distance Metric**: Euclidean distance
- **Training Split**: 80% train, 20% test

### Why Color Histograms?

✅ **Advantages**:
- Fast computation
- Rotation/scale invariant
- Simple implementation
- Good for color-dominant scenes (beaches, forests)

⚠️ **Limitations**:
- No spatial information
- Sensitive to lighting changes
- Limited feature richness

## 📈 Results Analysis

### Class Performance

**Beach (Class1)**:
- Higher precision (89%) - Few false positives
- Lower recall (78%) - Some beaches misclassified as forests
- Likely due to green vegetation in beach images

**Forest (Class2)**:
- High recall (93%) - Excellent at detecting forests
- Good precision (86%) - Mostly accurate predictions
- Color patterns more distinctive

### Confusion Matrix

```
                Predicted
              Beach  Forest
Actual Beach    40     11
      Forest     5     65
```

## 🛠️ Customization

### Change Number of Histogram Bins

Edit `extract_features.py`:
```python
def extract_features(image_path, bins=64):  # Change bins value
```

### Adjust KNN Parameters

Edit `train_model.py`:
```python
knn = KNeighborsClassifier(n_neighbors=3)  # Change k value
```

### Add More Classes

1. Create new folder in `dataset/`
2. Add class name to `classes` list
3. Re-run feature extraction and training

## 🔮 Future Improvements

- [ ] Add more image classes (mountains, cities, deserts)
- [ ] Implement deep learning (CNN) for better accuracy
- [ ] Add data augmentation (rotation, flipping, brightness)
- [ ] Create web interface for live predictions
- [ ] Use pre-trained models (ResNet, VGG)
- [ ] Add spatial features (HOG, SIFT)
- [ ] Implement cross-validation
- [ ] Add model hyperparameter tuning

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

Created as a computer vision and machine learning learning project.

## 🙏 Acknowledgments

- **OpenCV** - Computer vision library
- **scikit-learn** - Machine learning tools
- **icrawler** - Image downloading
- **Bing Images** - Dataset source

## 📧 Contact

For questions or suggestions, please open an issue in the GitHub repository.

---

⭐ **Star this repo if you find it helpful!**
