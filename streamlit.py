import streamlit as st
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from torchvision import transforms, models
from PIL import Image

st.title("Real-time ASL Detection App")

#Load YOLO hand detector
hand_model = YOLO("runs/detect/hand-detection13/weights/best.pt")
#hand_model = YOLO("runs/detect/hand-detector-scratch9-continued/weights/best.pt")

#Load ResNet sign classifier
sign_model = models.resnet18()
sign_model.fc = torch.nn.Linear(sign_model.fc.in_features, 24)
sign_model.load_state_dict(torch.load("resnet18_sign_classifier.pth", map_location="mps"))
#sign_model.load_state_dict(torch.load("resnet18_sign_classifier_scratch.pth", map_location="mps"))
sign_model.eval()

#Preprocessing transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

#Correct label map 
label_map = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K',
    10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P',
    15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U',
    20: 'V', 21: 'W', 22: 'X', 23: 'Y'
}

#Classify a cropped hand image
def classify_hand(image_np):
    image_pil = Image.fromarray(image_np).convert("L")
    image_tensor = transform(image_pil).unsqueeze(0)

    with torch.no_grad():
        outputs = sign_model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
        predicted_class = predicted.item()
        confidence_score = confidence.item()
        return label_map[predicted_class], confidence_score

#Turns of webcam
run = st.checkbox("Start Webcam")
FRAME_WINDOW = st.image([])
PREDICTION_TEXT = st.empty()

if run:
    cap = cv2.VideoCapture(0)
    while run:
        ret, frame = cap.read()
        if not ret:
            break

        #Detect hands
        results = hand_model(frame, verbose=False)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                #Crop directly using original box (no padding)
                hand_img = frame[y1:y2, x1:x2]

                try:
                    hand_resized = cv2.resize(hand_img, (224, 224))
                    predicted_sign, confidence_score = classify_hand(hand_resized)

                    #Draw box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, predicted_sign, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    #Show result in Streamlit
                    PREDICTION_TEXT.markdown(
                        f"### Predicted Sign: **{predicted_sign}**  \nConfidence: **{confidence_score * 100}%**"
                    )

                except:
                    pass

        #Show live frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb)

    cap.release()
