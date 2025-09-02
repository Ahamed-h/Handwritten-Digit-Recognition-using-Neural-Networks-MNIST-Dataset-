# Handwritten-Digit-Recognition-using-Neural-Networks-MNIST-Dataset-

# 🖊️ Handwritten Digit Recognition (MNIST)

This project implements a **Handwritten Digit Recognition System** using the **MNIST dataset**.  
It compares two deep learning approaches:
- **Fully Connected Neural Network (Dense)**
- **Convolutional Neural Network (CNN)**

Both models are trained, evaluated, and tested on **custom handwritten images** inside a **Jupyter Notebook / Google Colab**.

---

## 🚀 Features
- Preprocessing of input images with **OpenCV**.
- Implementation of **Dense NN** and **CNN** using TensorFlow/Keras.
- Model saving and loading using `.keras` format.
- Custom prediction on uploaded digit images.
- Training visualization with **Epoch vs Accuracy/Loss** plots using Matplotlib.

---

## 🛠️ Tech Stack
- **Python 3.x**
- **TensorFlow / Keras**
- **NumPy**
- **OpenCV**
- **Matplotlib**
- **Jupyter Notebook / Google Colab**

---

## 📊 Results
- **Dense Neural Network**: ~97% accuracy  
- **CNN Model**: ~99% accuracy  
- CNN generalizes better to unseen handwritten digits.

---

## 📂 How to Run (Google Colab)
1. Open the notebook in Colab:  
   [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/mnist-handwritten-digit-recognition/blob/main/mnist_nn_cnn.ipynb)

2. Install dependencies (if needed):
   ```python
   !pip install tensorflow opencv-python matplotlib
   ```
