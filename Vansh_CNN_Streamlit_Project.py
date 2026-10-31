# MINOR PROJECT 7: CNN IMAGE CLASSIFICATION AND STREAMLIT APPLICATION
# Student Name: Vansh Modi
# Part A (CNN) and Part B (Streamlit App)


import os
import sys
import numpy as np
import matplotlib.pyplot as plt


# PART A: CNN IMAGE CLASSIFICATION (MNIST DATASET)


def run_part_a_cnn():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, random_split

    print("PART A: CNN IMAGE CLASSIFICATION (MNIST DATASET)")

    # 1. Understand the image shape, classes, and sample count
    print("\n Step 1: Dataset Information ")
    transform = transforms.Compose([transforms.ToTensor()])
    
    full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    class_names = [str(i) for i in range(10)]
    print(f"Training Sample Count : {len(full_train_dataset)}")
    print(f"Testing Sample Count  : {len(test_dataset)}")
    print(f"Single Image Shape    : (1, 28, 28) [1 Grayscale Channel, 28x28 pixels]")
    print(f"Number of Classes     : 10 -> {class_names}")

    # Save Sample Images for PDF Report
    plt.figure(figsize=(8, 4))
    for i in range(10):
        img, label = full_train_dataset[i]
        plt.subplot(2, 5, i + 1)
        plt.imshow(img.squeeze().numpy(), cmap='gray')
        plt.title(f"Class: {label}")
        plt.axis('off')
    plt.suptitle("Sample MNIST Dataset Images (0 to 9)")
    plt.tight_layout()
    plt.savefig("1_sample_images.png")
    plt.close()
    print("-> Saved '1_sample_images.png' for PDF Report.")

    # 2. Preprocess the images and labels
    print("\n Step 2: Preprocessing & Train/Validation Split ")
    print("Pixel values normalized from [0, 255] to [0.0, 1.0].")
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    print(f"Train samples: {train_size} | Validation samples: {val_size} | Test samples: {len(test_dataset)}")

    # 3. Create and train a CNN model
    print("\n Step 3: CNN Model Architecture & Summary ")
    class MNIST_CNN(nn.Module):
        def __init__(self):
            super(MNIST_CNN, self).__init__()
            self.conv_layers = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.fc_layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 5 * 5, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 10)
            )

        def forward(self, x):
            x = self.conv_layers(x)
            x = self.fc_layers(x)
            return x

    device = torch.device("cpu")
    model = MNIST_CNN().to(device)
    
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Trainable Parameters: {total_params:,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    train_acc_list, val_acc_list = [], []
    train_loss_list, val_loss_list = [], []

    print("\n Training CNN Model (5 Epochs) ")
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation evaluation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] -> Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

    # Save Training & Validation Accuracy/Loss Graphs
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_acc_list, label='Training Accuracy', marker='o')
    plt.plot(range(1, epochs + 1), val_acc_list, label='Validation Accuracy', marker='s')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_loss_list, label='Training Loss', marker='o')
    plt.plot(range(1, epochs + 1), val_loss_list, label='Validation Loss', marker='s')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("2_training_curves.png")
    plt.close()
    print("-> Saved '2_training_curves.png' for PDF Report.")

    # 4. Evaluate the model using test data
    print("\n Step 4: Evaluating Model on Unseen Test Data ")
    model.eval()
    test_loss, test_correct, test_total = 0.0, 0, 0
    all_preds, all_labels, all_images = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images_dev, labels_dev = images.to(device), labels.to(device)
            outputs = model(images_dev)
            loss = criterion(outputs, labels_dev)
            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels_dev).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_images.extend(images.squeeze().numpy())

    final_test_loss = test_loss / test_total
    final_test_acc = test_correct / test_total
    print(f"Final Test Accuracy : {final_test_acc * 100:.2f}%")
    print(f"Final Test Loss     : {final_test_loss:.4f}")

    # 5. Show sample actual and predicted labels (Correct & Incorrect)
    print("\n Step 5 & 6: Generating Sample Predictions & Explanation ")
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_images = np.array(all_images)

    correct_idx = np.where(all_preds == all_labels)[0][:5]
    incorrect_idx = np.where(all_preds != all_labels)[0][:5]

    plt.figure(figsize=(10, 4))
    for i, idx in enumerate(correct_idx):
        plt.subplot(2, 5, i + 1)
        plt.imshow(all_images[idx], cmap='gray')
        plt.title(f"Actual: {all_labels[idx]}\nPred: {all_preds[idx]} (OK)", color='green', fontsize=9)
        plt.axis('off')
    for i, idx in enumerate(incorrect_idx):
        plt.subplot(2, 5, i + 6)
        plt.imshow(all_images[idx], cmap='gray')
        plt.title(f"Actual: {all_labels[idx]}\nPred: {all_preds[idx]} (ERR)", color='red', fontsize=9)
        plt.axis('off')
    plt.suptitle("Top Row: Correct Predictions | Bottom Row: Incorrect Predictions")
    plt.tight_layout()
    plt.savefig("3_predictions.png")
    plt.close()
    print("-> Saved '3_predictions.png' for PDF Report.")
    print("Explanation: The CNN achieves ~99% accuracy on test images. Rare errors occur when handwritten digits are ambiguous or incomplete.")


# PART B: STREAMLIT APPLICATION (STUDENT MARKS & GRADE CALCULATOR)


def run_part_b_streamlit():
    import streamlit as st
    import pandas as pd

    st.set_page_config(page_title="Student Marks Calculator", layout="centered")
    
    # Clear title and instructions
    st.title("Student Academic Marks & Grade Calculator")
    st.markdown("""
    **Instructions:** 
    1. Enter the student's full name.
    2. Enter marks obtained in each of the 5 subjects (must be between **0 and 100**).
    3. Click **Calculate Final Result** to view total marks, percentage, grade, and pass/fail status.
    """)

    st.divider()

    # User Inputs
    student_name = st.text_input("Student Full Name", placeholder="e.g., Vansh")

    col1, col2 = st.columns(2)
    with col1:
        math = st.number_input("Mathematics (0 - 100)", value=88.0, step=1.0)
        science = st.number_input("Science (0 - 100)", value=82.0, step=1.0)
        english = st.number_input("English (0 - 100)", value=79.0, step=1.0)
    with col2:
        computer = st.number_input("Computer Science (0 - 100)", value=94.0, step=1.0)
        history = st.number_input("History (0 - 100)", value=85.0, step=1.0)

    # Processing, Invalid-Input Handling, and Output
    if st.button("Calculate Final Result", type="primary"):
        marks_list = [math, science, english, computer, history]

        # Basic invalid-input handling
        if not student_name.strip():
            st.error("Invalid Input Error: Student Full Name cannot be empty. Please enter a valid name.")
        elif any(m < 0 or m > 100 for m in marks_list):
            st.error("Invalid Input Error: All subject marks must be strictly between 0 and 100.")
        else:
            total_marks = sum(marks_list)
            percentage = (total_marks / 500.0) * 100

            if percentage >= 90:
                grade, status = "A+ (Outstanding)", "PASS"
            elif percentage >= 80:
                grade, status = "A (Excellent)", "PASS"
            elif percentage >= 70:
                grade, status = "B (Good)", "PASS"
            elif percentage >= 60:
                grade, status = "C (Average)", "PASS"
            elif percentage >= 40:
                grade, status = "D (Pass)", "PASS"
            else:
                grade, status = "F (Fail)", "FAIL"

            st.subheader(f"Academic Report Card: {student_name}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Score", f"{total_marks:.1f} / 500")
            c2.metric("Percentage", f"{percentage:.2f}%")
            c3.metric("Final Grade", grade)

            df_summary = pd.DataFrame({
                "Subject Name": ["Mathematics", "Science", "English", "Computer Science", "History"],
                "Marks Obtained": marks_list,
                "Maximum Marks": [100, 100, 100, 100, 100]
            })
            st.table(df_summary)

            if status == "PASS":
                st.success(f"Final Result: {status} — Congratulations {student_name}!")
            else:
                st.warning(f"Final Result: {status} — Minimum 40% required to pass.")

# MAIN EXECUTION ROUTER

if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        running_in_streamlit = get_script_run_ctx() is not None
    except Exception:
        running_in_streamlit = False

    if running_in_streamlit:
        run_part_b_streamlit()
    else:
        run_part_a_cnn()