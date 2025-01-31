# Public Detection Using YOLOv8

## 🚀 Project Overview
This project utilizes **YOLOv8** for real-time **public/people detection**, leveraging the power of deep learning for object detection in **crowded environments**. It provides a streamlined approach to detecting individuals in images and videos with **high accuracy and efficiency**.

## 📦 Installation
To get started, install the necessary dependencies:
```bash
pip install ultralytics
pip install -U ipywidgets
pip install opencv-python numpy matplotlib
```

## 🏗 Model Training & Inference

### 1️⃣ Training YOLOv8 Model
Ensure your dataset is structured correctly, then run:
```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
model.train(data="/kaggle/input/public-detection-dataset-for-yolov8/People Detection.v8i.yolov8/data.yaml", epochs=30) 
```

### 2️⃣ Running Inference on Images
```python
!yolo predict model=/kaggle/working/runs/detect/train3/weights/best.pt source='/kaggle/input/public-detection-dataset-for-yolov8/Test_img_2.jpg'
```

### 3️⃣ Visualizing Test Images
```python
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load the trained model
model = YOLO('/kaggle/working/runs/detect/train2/weights/best.pt')  # Replace with the path to your best.pt

# Specify the path to the image or directory of images you want to visualize
image_path = '/kaggle/input/public-detection-dataset-for-yolov8/Test_img_2.jpg'

# Perform inference and visualize results
results = model(image_path)

# Function to plot and display results
def display_results(result):
    # Plot the results
    plt.imshow(result.plot())
    plt.axis('off')
    plt.show()

# Check if results is a list and iterate over each result
if isinstance(results, list):
    for result in results:
        display_results(result)
else:
    display_results(results)
```

### 4️⃣ Changing Kernel Directory (if using Colab/Kaggle)
```python
import os
os.chdir('/kaggle/working')
```

## 📂 Exporting Results
To save results in a compressed format:
```python
import shutil
shutil.make_archive('output_results', 'zip', 'runs/detect/')
```

## 📊 Key Features
✅ **Custom YOLOv8 Training** for public detection  
✅ **Optimized for Real-Time Performance**  
✅ **High Accuracy on Crowded Environments**  
✅ **Runs on Google Colab, Kaggle, or Local Setup**  

## 🔥 Results & Visualization
The model successfully detects individuals in crowded images with minimal false positives. Sample detections:

![image](https://github.com/user-attachments/assets/b931488e-3117-404d-b6d6-830dcd0d40b7)
![confusion_matrix_30_16e8429c4780d9eacf6b](https://github.com/user-attachments/assets/03ccddaa-9e61-44da-a9f6-0cb8b1897eea)
![labels_1_f558f6de0b4b3bae60c3](https://github.com/user-attachments/assets/00013117-9465-4846-9a7c-8d611d4643b8)
![results_30_cce216576518dfdcf4b3](https://github.com/user-attachments/assets/76e39ee1-59be-4b4d-a61d-aa4fd0d3cbeb)




## 📜 License
This project is open-source and free to use under the MIT License.

---
✉️ **For queries, reach out via LinkedIn or GitHub!** 🚀

