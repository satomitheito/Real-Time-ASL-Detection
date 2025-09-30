---
title: Real Time ASL Detector 
author: Satomi Ito
format: html
---
# Dataset

File: data_import.ipynb

### Hand Detection 

The hand detection portion of this project uses the [Kaggle Hand Detection Dataset](https://www.kaggle.com/datasets/nomihsa965/hand-detection-dataset-vocyolo-format), which consists of 2,061 labeled images featuring hands in various positions, sizes, and orientations. The dataset includes annotations in both YOLO and VOC formats. The bounding boxes provided identify the location of hands within each image.

The dataset is split into 1,551 training images and 510 test images. Images are diverse in terms of background, lighting conditions, and hand positioning. The annotations are single-class, focusing solely on detecting hands.


### Sign Language MNIST

For the sign classification task, the project uses the [Sign Language MNIST datase](https://www.kaggle.com/datasets/datamunge/sign-language-mnist). This dataset is a replacement for the original MNIST digits dataset, adapted for American Sign Language letter recognition. It contains 28x28 grayscale images of hand gestures representing the ASL alphabet (excluding the letters J and Z, which involve motion and cannot be captured in a static frame).

The dataset consists of 27,455 training images and 7,172 test images, each labeled with one of 24 classes corresponding to ASL letters A–Y. The original images were collected from multiple users and backgrounds, then processed with grayscale conversion, cropping, and augmentation techniques such as rotation, contrast adjustment, and pixelation.


# Pretrained 

File: pre_trained.ipynb


resnet18_sign_classifier.pth


Pretrained models were utilized for both the hand detection and sign classification components of the project.


### Hand Detection 

For the hand detection task, the project used Ultralytics' YOLOv8n model with pretrained weights (`yolov8n.pt`). The model was fine-tuned using YOLO-format annotations.

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(
    data="hand.yaml",
    epochs=20,
    imgsz=640,
    batch=16,
    name="hand-detection",
    device="mps"
)
```

The pretrained model provided a structure for identifying hand regions in varied lighting, orientation, and background conditions. By initializing with pretrained weights, the model required fewer epochs than from scratch.

![Photo Example](runs/detect/predict/VOC2007_12.jpg)

### Sign Language Classification
For the classification of static ASL signs, a ResNet18 model pretrained on ImageNet was employed. The final classification layer was replaced to output 24 classes corresponding to ASL letters A–Y. Grayscale 28×28 images from the Sign Language MNIST dataset were transformed to 3-channel 224×224 format to match the expected ResNet input.

```python
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 24)
```

Preprocessing included resizing, grayscale-to-RGB expansion, normalization, and conversion to tensor format. 

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
```

The model was trained over 10 epochs using the Adam optimizer and CrossEntropyLoss. 

![Train Results](images/train_p.png)

Evaluation on the test set resulted in a test accuracy of 95.37%, with an average inference time of 0.13 milliseconds per image, demonstrating the efficiency of the model in real-time settings.

![Test Results](images/test_p.png)

# Scratch

File: scratch.ipynb


resnet18_sign_classifier_scratch.pth


I also included a notebook where both the hand detection and sign language classification components were trained entirely from scratch.

### Hand Detection 

For hand detection, a YOLOv8n model was trained from scratch. Instead of using pretrained weights, the model was initialized with random weights using the YOLOv8n architecture configuration. 

```python
model = YOLO('yolov8n.yaml') 

model.train(
     data="hand.yaml",
     epochs=20,
     imgsz=640,
     batch=16,
     name="hand-detector-scratch",  
     save=True,                  
     save_period=1,            
     patience=20,              
     device = "mps"
    )
```

Initially, the model was trained for 20 epochs. While it learned basic hand localization, further evaluation, the streamlit deployment, suggested that 20 epochs were not sufficient, especially when trained from scratch. Therefore, training was resumed from the final weights of the first run (last.pt) for an additional 40 epochs, bringing the total to 60 epochs.

```python
model = YOLO("runs/detect/hand-detector-scratch9/weights/last.pt")

model.train(
    data="hand.yaml",
    epochs=40,                
    batch=16,
    imgsz=640,
    name="hand-detector-scratch9-continued",
    device="mps"
)
```

![Photo Example](runs/detect/predict2/VOC2007_24.jpg)

### Sign Language Classification
For static ASL sign classification, a ResNet18 model was initialized with random weights (weights=None). The same preprocessing pipeline was used as in the pretrained version.

```python
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 24)
```

Training was performed over 10 epochs using the Adam optimizer and cross-entropy loss.

![Train Results](images/train_s.png)

The test accuracy for the scratch-trained ResNet18 was 94.80%, slightly below the pretrained version’s 95.37%.

![Test Results](images/test_s.png)

# Streamlit Deployment

File: streamlit.py


The application processes video frames from the user's webcam in real time using the following pipeline:

- **Hand Detection**: A YOLOv8 model trained from scratch is used to detect hands in each frame. The model outputs bounding box coordinates around the detected hand.
- **Cropping**: The hand region is cropped from the frame using the bounding box, and optionally expanded for better context.
- **Preprocessing**: The cropped image is resized to 224×224 pixels and converted to a 3-channel format, matching the input requirements of the ResNet18 classifier.
- **Sign Classification**: The processed image is passed through a ResNet18 model (either pretrained or trained from scratch) to classify the static ASL sign.
- **Display**: The predicted sign and its confidence score are displayed in the app, with bounding boxes drawn over the original video feed.

![Streamlit App](images/me.png)


