![image](https://github.com/user-attachments/assets/6db6e21d-fea9-4846-b03a-9f6612efdf5d)# 1mao-chinese-coin
This study is a system for detecting and classifying the front and back sides of Chinese 1 Mao coins using YOLOv8 and Python GUI (Tkinter). Includes image/video input, object detection, and result export to annotated images and CSV.
---

## 🎯 Key Features

- 🔎 **Coin Detection**: Uses a custom-trained YOLOv8 model to identify coins.
- 🧮 **Automatic Counting**: Detects and counts each coin type in the image.
- 📷 **Visual Feedback**: Shows bounding boxes information and confidence scores in csv file.
- 🚀 **Tkinter UI**: Fast and basic interface.

---
## 💶 Supported 1mao Chinese Coin
Front : 0
Back: 1
---
## 🚀 Getting Started

### 📦 Requirements

- Python 3.8+
- pip

### 🛠 Installation

1. Clone the repository:

```bash
git clone https://github.com/VuthilinhVn/1mao-chinese-coin 
cd Euro-Coin-Counting-App
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```
3. Make sure your YOLOv8 model weights are in:

```
coin_count/weights/best.pt
```

---

## 🖥️ Running the App

```bash
python app.py
```

or double-click the packaged app.exe file to run the application.

---


