
# ISBAK Smart Intersection and Emergency Vehicle Prioritization System

This repository contains a computer vision-based smart intersection management and emergency vehicle prioritization system developed as part of the İSBAK internship project. The system utilizes deep learning to detect vehicles and dynamically prioritize emergency response units.

## 🚀 Features

* **Vehicle Detection & Classification:** Detects standard vehicles as well as emergency response units using a custom-trained YOLOv8 model.
* **Emergency Prioritization:** Identifies ambulances, police cars, and fire trucks to optimize traffic flow and signal management.
* **Real-time Processing:** Processes video streams efficiently with OpenCV and object tracking algorithms.

---

## 📁 Project Structure

```text
isbak_proje1/
│
├── Akilli_kavsak_final.py    # Main application script
├── best.pt                   # Custom trained YOLOv8 model weights
├── test_ambulans.mp4         # Ambulance test video
├── test_polis.mp4            # Police car test video
└── test_itfaiye.mp4          # Fire truck test video

```

---

## 🛠️ Tech Stack & Libraries

* **Python** (3.8+)
* **Ultralytics YOLOv8**
* **OpenCV**

---

## ⚙️ Installation & Requirements

1. Clone the repository:
```bash
git clone [https://github.com/ysresraaa/ISBAK-Smart-Intersection-System.git](https://github.com/ysresraaa/ISBAK-Smart-Intersection-System.git)

```


2. Install the required dependencies:
```bash
pip install ultralytics opencv-python

```



---

## ▶️ Usage

Run the main script to start the smart intersection system:

```bash
python Akilli_kavsak_final.py

```

```

```
