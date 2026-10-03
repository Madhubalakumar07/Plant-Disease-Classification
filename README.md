# 🌿 Plant Disease Classification using Deep Learning

A deep learning-based plant disease classification system that identifies plant diseases from leaf images using a custom Convolutional Neural Network (CNN) built with **PyTorch**.

The trained model is integrated with a **Streamlit web application**, allowing users to upload a plant leaf image and receive the predicted disease class along with the model's confidence score.

---

## 🚀 Live Demo

🌐 **Streamlit Application:**  
https://plant-disease-classification-madhu0725.streamlit.app/

---

## 📌 Project Overview

Plant diseases can significantly affect crop health and agricultural productivity. Early identification of diseases can help in taking appropriate action before the disease spreads further.

This project uses **Computer Vision and Deep Learning** to classify plant leaf images into different disease categories.

The complete system consists of:

- PlantVillage dataset
- Image preprocessing
- Data augmentation
- Custom CNN architecture
- PyTorch model training
- Model evaluation
- Trained model checkpoint
- Streamlit web application
- Real-time image classification

### 🔄 Project Workflow

```text
                    Plant Leaf Image
                           │
                           ▼
                  Image Preprocessing
                           │
                           ▼
                    CNN Model
                           │
                           ▼
                  Feature Extraction
                           │
                           ▼
                    Classification
                           │
                           ▼
                  Predicted Disease
                           │
                           ▼
                   Confidence Score
```

---

## ✨ Features

- 🌱 Plant leaf disease classification
- 🧠 Custom Convolutional Neural Network
- 🔥 PyTorch-based deep learning model
- 📷 Supports JPG, JPEG and PNG images
- ⚡ Real-time prediction
- 📊 Prediction confidence score
- 🖥️ GPU support during training/inference when available
- 🌐 Streamlit web interface
- ☁️ Streamlit Community Cloud deployment
- 📦 Model checkpoint containing trained weights and class information

---

# 🗂️ Dataset

This project uses the **PlantVillage dataset**.

### Dataset Source

[Kaggle - PlantVillage Dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)

The dataset contains images of plant leaves belonging to different healthy and diseased categories.

The model used in this project contains **15 classes**.

### 🌶️ Pepper

1. Pepper Bell Bacterial Spot
2. Pepper Bell Healthy

### 🥔 Potato

3. Potato Early Blight
4. Potato Healthy
5. Potato Late Blight

### 🍅 Tomato

6. Tomato Bacterial Spot
7. Tomato Early Blight
8. Tomato Healthy
9. Tomato Late Blight
10. Tomato Leaf Mold
11. Tomato Septoria Leaf Spot
12. Tomato Spider Mites
13. Tomato Target Spot
14. Tomato Mosaic Virus
15. Tomato Yellow Leaf Curl Virus

> The dataset is not included in this GitHub repository because of its large size. The dataset directory is excluded using `.gitignore`.

---

# 🧠 Deep Learning Model

The project uses a custom **Convolutional Neural Network (CNN)** implemented using PyTorch.

The model contains **5 convolutional layers** followed by fully connected layers for final classification.

## 🏗️ Model Architecture

```text
Input Image
     │
     ▼
Resize to 256 × 256
     │
     ▼
Conv2D: 3 → 32
     │
     ▼
ReLU
     │
     ▼
Max Pooling
     │
     ▼
Conv2D: 32 → 64
     │
     ▼
ReLU
     │
     ▼
Max Pooling
     │
     ▼
Conv2D: 64 → 128
     │
     ▼
ReLU
     │
     ▼
Max Pooling
     │
     ▼
Conv2D: 128 → 256
     │
     ▼
ReLU
     │
     ▼
Max Pooling
     │
     ▼
Conv2D: 256 → 512
     │
     ▼
ReLU
     │
     ▼
Max Pooling
     │
     ▼
Flatten
     │
     ▼
Fully Connected
512 × 8 × 8 → 1024
     │
     ▼
ReLU
     │
     ▼
Fully Connected
1024 → 512
     │
     ▼
ReLU
     │
     ▼
Fully Connected
512 → 15
     │
     ▼
Disease Prediction
```

---

## 🔬 Convolutional Layers

| Layer | Input Channels | Output Channels | Kernel Size |
|------|---------------:|----------------:|------------:|
| Conv1 | 3 | 32 | 3 × 3 |
| Conv2 | 32 | 64 | 3 × 3 |
| Conv3 | 64 | 128 | 3 × 3 |
| Conv4 | 128 | 256 | 3 × 3 |
| Conv5 | 256 | 512 | 3 × 3 |

Each convolutional layer is followed by:

- ReLU activation
- Max Pooling

---

# 🔄 Image Preprocessing

Images are resized to:

```text
256 × 256
```

The training pipeline uses image transformations such as:

- Image resizing
- Random rotation
- Random horizontal flipping
- Conversion to PyTorch tensors

For inference, the uploaded image is transformed using:

```python
transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
```

The processed image is then converted into a batch tensor before being passed to the model.

---

# 🏋️ Model Training

The model was trained using **PyTorch**.

The training pipeline follows:

```text
PlantVillage Dataset
        │
        ▼
     ImageFolder
        │
        ▼
Image Transformations
        │
        ▼
Train / Validation / Test Split
        │
        ▼
      DataLoader
        │
        ▼
       CNN
        │
        ▼
   Forward Pass
        │
        ▼
    Loss Calculation
        │
        ▼
  Backpropagation
        │
        ▼
     Optimizer
        │
        ▼
    Model Update
        │
        ▼
      Evaluation
```

### Training Augmentation

The training images are augmented using transformations such as:

```text
Random Rotation
Random Horizontal Flip
Resize
Tensor Conversion
```

These transformations help the model learn from slightly different variations of the training images.

---

# 📊 Model Evaluation

The model is evaluated using a separate test dataset.

The main evaluation metrics are:

- Test Loss
- Test Accuracy

### Final Results

```text
Test Loss     : XX.XXXX
Test Accuracy : XX.XX%
```

> Replace the values above with the metrics from the final training run used to generate the deployed model.

---

# 💾 Model Saving

The trained model is saved as a PyTorch checkpoint.

```python
torch.save({
    "model_state_dict": model.state_dict(),
    "class_names": classes,
    "num_classes": len(classes)
}, "plant_disease_model.pth")
```

The checkpoint contains:

```text
plant_disease_model.pth
│
├── model_state_dict
├── class_names
└── num_classes
```

The model architecture is defined separately in `model.py`, allowing the application to recreate the architecture and load the trained weights without retraining.

---

# 🌐 Streamlit Application

The trained PyTorch model is integrated into a Streamlit web application.

The application allows users to:

1. Upload a plant leaf image
2. Preview the uploaded image
3. Resize and preprocess the image
4. Pass the image through the trained CNN
5. Calculate class probabilities
6. Identify the predicted disease
7. Display the confidence score

The prediction probabilities are calculated using Softmax:

```python
probabilities = torch.softmax(output, dim=1)
```

The class with the highest probability is selected as the prediction.

```text
Uploaded Image
      │
      ▼
Preprocessing
      │
      ▼
PyTorch CNN
      │
      ▼
Softmax
      │
      ▼
Highest Probability
      │
      ▼
Predicted Disease
      │
      ▼
Confidence %
```

---

# 🛠️ Technologies Used

## Programming Language

- Python

## Deep Learning

- PyTorch
- Torchvision

## Image Processing

- Pillow

## Data Processing

- NumPy
- Pandas

## Web Application

- Streamlit

## Development Tools

- Jupyter Notebook
- Visual Studio Code
- Git
- GitHub

## Deployment

- Streamlit Community Cloud

---

# 📁 Project Structure

```text
plant-disease-classification/
│
├── app.py
├── model.py
│
├── Model/
│   └── plant_disease_model.pth
│
├── training/
│   └── training.ipynb
│
├── datasets/
│   └── PlantVillage/
│
├── requirements.txt
├── runtime.txt
├── .gitignore
└── README.md
```

### Important Files

| File | Description |
|------|-------------|
| `app.py` | Streamlit application |
| `model.py` | CNN architecture |
| `plant_disease_model.pth` | Trained PyTorch model |
| `training.ipynb` | Model training notebook |
| `requirements.txt` | Python dependencies |
| `runtime.txt` | Python runtime configuration |
| `.gitignore` | Files excluded from Git |
| `README.md` | Project documentation |

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Madhubalakumar07/plant-disease-classification.git
```

Move into the project directory:

```bash
cd plant-disease-classification
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the Streamlit Application

```bash
streamlit run app.py
```

---

# 📦 Requirements

The main dependencies used by the application are:

```text
streamlit
torch
torchvision
pillow
```

Additional packages required for model training can be installed depending on the training notebook.

---

# 🚀 Deployment

The application can be deployed using **Streamlit Community Cloud**.

```text
GitHub Repository
       │
       ▼
Streamlit Community Cloud
       │
       ▼
Install requirements
       │
       ▼
Load PyTorch Model
       │
       ▼
Start Streamlit App
       │
       ▼
Public Web Application
```

Live application:

https://plant-disease-classification-madhu0725.streamlit.app/

---

# 📦 Large Model File

The trained `.pth` model is relatively large.

GitHub has a **100 MB limit for individual files in normal Git repositories**.

For a model larger than this limit, **Git Large File Storage (Git LFS)** can be used.

```bash
git lfs install
git lfs track "Model/plant_disease_model.pth"
git add .gitattributes
git add Model/plant_disease_model.pth
git commit -m "Add trained plant disease model"
git push
```

The dataset itself should remain excluded from Git.

---

# 🔐 `.gitignore`

The dataset can be excluded using:

```gitignore
datasets/
```

Other generated and environment-specific files can also be ignored:

```gitignore
__pycache__/
*.py[cod]
.ipynb_checkpoints/

venv/
.venv/
env/

.env

.vscode/

*.log

.DS_Store
Thumbs.db
```

If Git LFS is being used for the model, do **not** ignore the model with:

```gitignore
*.pth
```

unless the specific model file is explicitly unignored.

---

# 🔮 Future Improvements

## 🧠 Model Improvements

- Transfer learning using ResNet
- EfficientNet-based classification
- MobileNet for lightweight deployment
- Deeper CNN architectures
- Hyperparameter tuning
- Learning-rate scheduling
- Early stopping
- Class imbalance handling
- Model quantization
- Model pruning
- Faster inference

## 📊 Prediction Improvements

- Top-3 predictions
- Probability distribution chart
- Confusion matrix
- Per-class precision
- Per-class recall
- F1-score
- Detailed classification report

## 🌱 Application Improvements

- Disease descriptions
- Disease symptoms
- Preventive measures
- Treatment information
- Plant health recommendations
- Camera-based image capture
- Prediction history
- Multiple image uploads
- Mobile-friendly UI

## ☁️ Deployment Improvements

- Docker deployment
- REST API
- Optimized model serving
- Model versioning
- Cloud-based inference
- Model compression

---

# ⚠️ Limitations

This project is primarily intended as a **machine learning and computer vision project**.

The model's prediction can be affected by:

- Image quality
- Lighting conditions
- Leaf orientation
- Background noise
- Image resolution
- Unseen diseases
- Diseases not represented in the dataset
- Differences between dataset images and real-world field images

The PlantVillage dataset contains controlled image conditions that may differ from real-world agricultural environments.

Therefore, predictions from this application should not be treated as a definitive agricultural diagnosis.

---

# 📚 Dataset Reference

**PlantVillage Dataset**

Kaggle:

https://www.kaggle.com/datasets/arjuntejaswi/plant-village

---

# 🎯 Learning Objectives

This project helped implement and understand:

- Image classification
- Convolutional Neural Networks
- PyTorch
- Torchvision
- Image preprocessing
- Data augmentation
- Dataset splitting
- DataLoader
- Model training
- Loss calculation
- Backpropagation
- GPU acceleration
- Model evaluation
- Model serialization
- Streamlit deployment
- Git and GitHub
- Git LFS

---

# 👨‍💻 Author

## Madhubalakumar S.

**B.Tech Artificial Intelligence and Machine Learning**  
**Bannari Amman Institute of Technology**

### Profiles

- GitHub: https://github.com/Madhubalakumar07
- LinkedIn: https://www.linkedin.com/in/madhubalakumar-s-9a4b00329/
- LeetCode: https://leetcode.com/u/Madhubalakumar/
- HackerRank: https://www.hackerrank.com/madhubalakumars1/

---

# ⭐ Support

If you found this project useful, consider giving the repository a ⭐ star.

---

# 📄 License

This project is intended for educational and research purposes.

The PlantVillage dataset is obtained from Kaggle. Please refer to the original dataset source and its applicable license and terms before redistributing or using the dataset.

---

## 🌿 Plant Disease Classification

**Built with Python, PyTorch, Computer Vision and Streamlit.**
