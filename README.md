# Minor Project 7: CNN Image Classification & Streamlit Application

**Student Name:** Vansh Modi
**Dataset:** MNIST Handwritten Digits (0–9)  
**Streamlit App Topic:** Student Academic Marks & Grade Calculator  

---

## Project Overview

This repository contains the complete implementation and documentation for **Minor Project 7**, divided into two main parts inside a single Python script:

### Part A — CNN Image Classification (MNIST Dataset)
- **Dataset:** 70,000 grayscale images (`28x28` pixels) of handwritten digits from `0` to `9` (60,000 training / 10,000 testing).
- **Preprocessing:** Pixel normalization to `[0.0, 1.0]` and a 90:10 training-validation split.
- **Model Architecture:** PyTorch Convolutional Neural Network (`MNIST_CNN`) featuring two convolutional + max-pooling blocks (`32` and `64` filters), followed by a flattened dense classifier (`128` hidden units with `30%` dropout, `10` output classes).
- **Performance:** Achieves **~99.0% test accuracy** after 5 epochs using the `Adam` optimizer and `CrossEntropyLoss`.

### Part B — Streamlit Application (Student Marks & Grade Calculator)
- **Purpose:** An interactive web application that computes a student's total score, percentage, academic grade, and Pass/Fail status across 5 core subjects.
- **Features:**
  - Clean two-column numeric input layout for subject scores (`0–100`).
  - Built-in input validation (`st.error`) for empty student names or out-of-range marks.
  - Instant summary KPI metric cards, tabular report card breakdown, and status banners.

---

## Repository Files

- `Vansh_CNN_Streamlit_Project.py` — Single Python file containing both **Part A (CNN workflow)** and **Part B (Streamlit app)**.
- `UNLOX MINOR PROJECT 7 REPORT.pdf` — Complete project report containing dataset details, model summary, accuracy/loss curves, sample predictions, and Streamlit screenshots.
- `1_sample_images.png` — Sample images from the MNIST dataset.
- `2_training_curves.png` — Training and validation accuracy and loss curves.
- `3_predictions.png` — Sample correct and incorrect test set predictions.

---

## How to Run Locally

### 1. Install Dependencies
pip install "torch==2.2.2" "torchvision==0.17.2" "numpy<2.0" "pyarrow<16.0.0" streamlit matplotlib pandas

### 2. Run Part A (CNN Training & Evaluation)
python Vansh_CNN_Streamlit_Project.py

### 3. Run Part B (Streamlit Web Application)
streamlit run Vansh_CNN_Streamlit_Project.py
