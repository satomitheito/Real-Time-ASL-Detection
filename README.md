# Mini-project-3: Real-time Sign Language Detection Application

**Goal**: Develop a real-time application that detects and interprets sign language digits using a combination of object detection and classification models. The project will involve building a pipeline that detects hands in video frames and classifies the detected hand gestures into sign language digits.

## Project Summary

### Group Selection

You can work alone if you want but recommended to work as a group

- **Form a group of 3 people**
  - (Optional): Select a group leader
  - Plan out your project in a systematic way 
  - Break the project into stages and allocate sub-components to individuals 
  - Feel free to use Cursor, or similar, for coding acceleration 
  - Use Git, branches, etc to collaborate professionally, and efficiently 

### Reading & Research:

To learn more about the technologies involved, see the following resources:

- [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/)
- [Vision Transformers](https://arxiv.org/abs/2010.11929)
- [Sign Language MNIST Dataset](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)

### Data Acquisition:

- Start simple!!! Use the provided datasets to debug and develop your pipeline. Once the project is working, you can apply a similar workflow to a different task with another dataset if desired.

`Provided Data`

- Hand Detection Dataset: [Kaggle Hand Detection Dataset](https://www.kaggle.com/datasets/nomihsa965/hand-detection-dataset-vocyolo-format)
- Sign Language Dataset: [Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)

`Exploration:` 

Consider applying the workflow to a different task with a different dataset if you wish to explore beyond the initial scope. The theme is to perform real-time object detection followed by another interesting task. For those interested in extending the project further, you could explore using advanced techniques such as vision transformers to enhance the detection and classification capabilities.

### Preprocessing:

- **Video Frames:** Extract frames from video input and preprocess them for hand detection.
- **Images:** Standardize image sizes (e.g., resize to 224×224) and perform necessary augmentations.
- As usual, have a training/test/validation split.

### Model Training:

- **Hand Detection:** Train a YOLO model to detect hands in video frames.
- **Sign Language Classification:** Train models (e.g., ResNet architectures) to classify hand gestures into sign language digits.
- Experiment with both transfer learning (pretrained models) and training from scratch where possible.
- Compare performance metrics (accuracy, loss, inference time) across models and modalities.
- Tune the models as needed for best results.

### Deployment:

- Create an interactive application (e.g., using Streamlit) that allows users to:
  - Upload a video or use a webcam to detect and classify sign language gestures in real-time.
  - Display both detection and classification results with model confidence scores.

### Documentation & Submission:

- Submit to the code repository with clear documentation and comments.
- Provide a brief report summarizing your approach, data preprocessing steps, model comparisons, challenges encountered, and final evaluation metrics.
* After two weeks, you will need to submit the following:
  * A link to your GitHub code repository 
  * Make a slide deck (or jupyter notebook), to describe what you did and showcase your results (submit PDF or HTML to Canvas)
  * Do a recording where you walk through and describe your results and presentation (10-20 minutes) (submit the recording to Canvas presentation recording)
  * At the end of the mini-project, a group will be selected at random to do their presentation "live" in front of class (to fuel discussion)

## Appendix

### Dataset Information 

More information on the provided datasets:

#### Hand Detection

The dataset includes images annotated for hand detection, suitable for training YOLO models.

#### Sign Language

The Sign Language MNIST dataset includes images of hand gestures representing digits 0-9 in American Sign Language.

**Inspiration**: How accurately can we interpret sign language gestures in real-time using video input?

**Source**: [Kaggle Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
