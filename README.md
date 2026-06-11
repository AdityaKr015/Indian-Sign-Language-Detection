# **Indian-Sign-Language-Detection**

**Frameworks & Libraries:- **
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv11-00FFFF)

**Platforms:- **
![Roboflow](https://img.shields.io/badge/Roboflow-6100ee?logo=roboflow&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?logo=kaggle&logoColor=white)

**License:- **
![MIT](https://img.shields.io/badge/License-MIT-green)

A College Deeplearning Project build Sign Detection in Hindi langauge annotation.

## 📓 Training Notebook

The full model training pipeline can be found here:

[Open Notebook](notebook/Indian Sign Language Detection.ipynb)

## **Project Overview**

In this project, I build a real time hand sign detection which shows hindi labels with confidence score using transfer learning (pretrained yolo v11 model) using dataset from Roboflow and fine tune train on Kaggle platform.

The main goal of this project is to:-

- Helps to understand the sign langauge in hindi for people who don't know hand sign langauge.

## **Features**

- Real-time hand sign detection.
- Bounding Box labeled objects with confidence.
- Supports classification of 100+ classes.

## **Implementation**

Project implementation started with dataset preparation and ending with real time detection results.The entire workflow uses platform Kaggle for training and Roboflow Universe for data prepation.

### 1. **Dataset Integration**

- The dataset was used from Roboflow which consists on 110 classes with hindi annotations.
- Each image was annotated with bounding boxes marking the location of objects.
- The dataset was exported in YOLO compatible format (with training, validation, and test splits).
- The dataset was uploaded in my kaggle dataset section to load into jupyter notebook for model training.

### 2. **Environment Setup**
 
- In Kaggle, I installed the Ultralytics Library to use YOLO.
- Setup the dual GPU (T4  2) for training runtime.
-	Model was evaluted on val datasplit.
- Model ran on test datasplit to product bounding box with label, confidence score images.
-	Zipped the output folder to made it downloadable to save on local machine. 

### 3. **Model Training**
 
We fine-tuned the pre-trained YOLOv11m model from Ultralytics using transfer learning, optimized for real-time performance, speed, and accuracy on dual NVIDIA T4 GPUs.

- Training parameters:-

| Parameter |	Value |	Purpose |
|------|--------|--------|
|Epochs / Batch	|200 / 48	    |Stable convergence, optimized for dual T4 GPU memory |
|Img Size	      |640 × 640	  |Balanced trade-off between detail and inference speed |
|Optimizer	    |AdamW	      |Momentum 0.9, weight decay 0.0005 for regularization |
|Learning Rate	|0.001 → 0.01	|Cosine decay (cos_lr=True) with a 3-epoch warmup |
|Loss Weights	  |box=7.5, cls=1.5	| Tightens bounding boxes and handles class imbalance |

- Data Augmentation Strategy:- To ensure robustness against real-world lighting and camera angles, the following augmentations were applied:
  
  - Color (HSV):= Hue (±0.015), Saturation (±0.7), and Brightness (±0.4) variations.
  - Geometry:- Rotation (±10°), translation (±10%), scale/zoom (±50%), shear (±10°), perspective (0.001), and horizontal flip (50% probability).
  - Context:- Mosaic (1.0) combining 4 training images to improve small-object/hand gesture detection.

### 4. Model Validation
The model was evaluated on a dedicated validation split to assess generalization to unseen data. The key evaluation metrics used include:

- Precision: Percentage of predicted detections that were correct.
- Recall: Percentage of actual hand gestures successfully detected.
- mAP@50: Mean Average Precision calculated at an IoU (Intersection over Union) threshold of 0.50 (measures general detection accuracy).
- mAP@50-95: Mean Average Precision calculated across a range of IoU thresholds from 0.50 to 0.95 (measures localization precision and boundary tightness).

### 5. **Inference and Output**
 
- The trained YOLOv1 model was used for inference on new test images.
- Each detected object was marked with a bounding box, label, and confidence score.
- Results showed that the model successfully detected multiple objects simultaneously and delivered accurate predictions.
- The outputs demonstrated the system’s ability to be applied in real world scenarios such as surveillance, product detection, and automation tasks.

# **Confusion Matrix & Results data**

<img width="2400" height="1200" alt="image" src="https://github.com/user-attachments/assets/e5372bff-7512-4bd3-b11b-29b9910437fa" />

<img width="3000" height="2250" alt="image" src="https://github.com/user-attachments/assets/eee6ee2f-fd13-4fde-b2ff-855c88542d32" />


## **WORKFLOW:- **

```mermaid
flowchart TD
    A["🗄️ Dataset Source
    Roboflow Universe
    6K+ Labeled Images
    110 Hindi Classes"]

    B["📦 Dataset Preparation
    BBox Annotation
    YOLO Format Export
    Train / Val / Test Split"]

    C["☁️ Kaggle Platform
    Upload Dataset
    Setup Dual T4 GPU
    Install Ultralytics"]

    D["🧠 Pretrained Model
    YOLOv11m
    Ultralytics
    Transfer Learning"]

    E["⚙️ Model Training
    Epochs: 200 | Batch: 48
    Img Size: 640×640
    Optimizer: AdamW | LR: 0.001"]

    F["🎨 Data Augmentation
    HSV Color Jitter
    Rotation ±10° | Mosaic
    Flip / Scale / Shear"]

    G["📊 Model Validation
    Precision: 87.9%
    Recall: 88.0%
    mAP@50: 86.4%"]

    H{"✅ Accuracy
    Acceptable?"}

    I["💾 Save Best Weights
    best.pt
    Download via ZIP"]

    J["🔍 Inference & Testing
    Test Dataset
    Bounding Boxes
    Hindi Labels + Confidence"]

    K["🎯 Real-Time Detection
    Webcam / Video Feed
    Live Hindi Sign Labels
    Confidence Scores"]

    A --> B
    B --> C
    B --> D
    C --> E
    D --> E
    E --> F
    F --> G
    G --> H
    H -- No --> E
    H -- Yes --> I
    I --> J
    J --> K

    style A fill:#6100ee,color:#fff,stroke:#4a00b4,stroke-width:2px
    style B fill:#4A90D9,color:#fff,stroke:#2c6fad,stroke-width:2px
    style C fill:#20BEFF,color:#000,stroke:#0090cc,stroke-width:2px
    style D fill:#00CED1,color:#000,stroke:#009aa0,stroke-width:2px
    style E fill:#FF6B35,color:#fff,stroke:#cc4a10,stroke-width:2px
    style F fill:#E8A838,color:#000,stroke:#b87d10,stroke-width:2px
    style G fill:#2ECC71,color:#fff,stroke:#1a9e50,stroke-width:2px
    style H fill:#E74C3C,color:#fff,stroke:#b52a1c,stroke-width:2px
    style I fill:#8E44AD,color:#fff,stroke:#6a2585,stroke-width:2px
    style J fill:#3498DB,color:#fff,stroke:#1a6fad,stroke-width:2px
    style K fill:#27AE60,color:#fff,stroke:#1a7a40,stroke-width:2px
```

╔══════════════════════════════════════════════════════════════════════════════╗
║              INDIAN SIGN LANGUAGE DETECTION — WORKFLOW                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

  ┌─────────────────────┐
  │   📡 DATA SOURCE    │
  │   Roboflow Universe │
  │   6K+ Images        │
  │   110 Hindi Classes │
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐
  │  📦 DATA PREP       │
  │  BBox Annotation    │
  │  YOLO Format        │
  │  Train/Val/Test     │
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐     ┌─────────────────────┐
  │  ☁️ KAGGLE SETUP    │────▶│  🧠 YOLOv11m BASE   │
  │  Dual T4 GPU        │     │  Pretrained Weights  │
  │  Ultralytics Lib    │     │  Transfer Learning   │
  └──────────┬──────────┘     └──────────┬──────────┘
             │                           │
             └──────────┬────────────────┘
                        │
                        ▼
           ┌────────────────────────┐
           │   ⚙️ MODEL TRAINING    │
           │  Epochs    : 200       │
           │  Batch     : 48        │
           │  Img Size  : 640×640   │
           │  Optimizer : AdamW     │
           │  LR        : 0.001     │
           │  Loss Box  : 7.5       │
           └────────────┬───────────┘
                        │
                        ▼
           ┌────────────────────────┐
           │  🎨 AUGMENTATION       │
           │  HSV Color Jitter      │
           │  Rotation ±10°         │
           │  Mosaic (4 Images)     │
           │  Flip / Scale / Shear  │
           └────────────┬───────────┘
                        │
                        ▼
           ┌────────────────────────┐
           │  📊 VALIDATION         │
           │  Precision : 87.9%     │
           │  Recall    : 88.0%     │
           │  mAP@50    : 86.4%     │
           │  mAP@50-95 : 51.0%     │
           └────────────┬───────────┘
                        │
               ┌────────▼────────┐
               │  ✅ Acceptable? │
               └────────┬────────┘
                  NO ◀──┘──▶ YES
                  │             │
                  │             ▼
                  │   ┌──────────────────┐
                  │   │  💾 SAVE WEIGHTS │
                  │   │  best.pt         │
                  │   │  Download ZIP    │
                  │   └────────┬─────────┘
                  │            │
                  └────────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │  🔍 INFERENCE          │
                  │  Test Images           │
                  │  Bounding Boxes        │
                  │  Hindi Labels          │
                  │  Confidence Scores     │
                  └────────────┬───────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │  🎯 REAL-TIME OUTPUT   │
                  │  Live Webcam Feed      │
                  │  Hindi Sign Labels     │
                  │  Instant Recognition   │
                  └────────────────────────┘
                  
## **Project Structure**
```
├── notebook/Indian Sign Language Detection.ipynb         # Main Jupyter Notebook

├── detect/                                               # Training & testing images

├── detect/train/weights                                  # Saved models 

├── requirements.txt                                      # Dependencies

└── README.md                                             # Project documentation
```

## ⚙️ **Tech Stack**

- `Language`:- Python
- `Frameworks/Libraries`:- Ultralytics (YOLO Framework)
- `Tools`:- Kaggle, Roboflow

## Dataset

### Dataset Source:-

https://universe.roboflow.com/mujab/sign-language-w9k46/dataset/1

Contains 6k of labeled images of sign language with hindi annotations.


## 📈 **Outputs**

<img width="640" height="640" alt="aaj_16_jpg rf f1cbd4850e0dfc9093eae2503a0d5cd9" src="https://github.com/user-attachments/assets/1de3c7a4-4bed-47bd-a8ad-fc04dda6ae41" />
<img width="640" height="640" alt="alvidah_5_jpg rf 2aa8ccbd69f6209ca0ed36947563eb8d" src="https://github.com/user-attachments/assets/b384f224-409c-4082-a836-fd45a29aedee" />
<img width="640" height="640" alt="acha_27_jpg rf 25754bccd89a77a440b0eb56a18ea96e" src="https://github.com/user-attachments/assets/300ac5d2-8eb4-4f3c-8bce-557c4eb80334" />
<img width="640" height="640" alt="baithna_51_jpg rf beb30d6de683cf3550da89ba338bc999" src="https://github.com/user-attachments/assets/4f02a642-eabb-45e8-9ae5-0781dee17178" />
<img width="640" height="640" alt="ghalat_49_jpg rf 9129856a7c4485f144a0f56cd76a5d99" src="https://github.com/user-attachments/assets/be39f059-2666-4699-a794-9372fb1ee43a" />


## Results:

### Evaluation Metrics:-

| Metrics | Score |
|-------|--------|
|`Precision`| ~ 87.9%|
|`Recall`| ~ 88.0%|
|`mAP@50`| ~ 86.4%|
|`mAP@50–95`:| ~ 51.0%|

Correctly classifies speed limits, stop signs, and other common traffic symbols.


 

