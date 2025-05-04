import sys
import os
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import csv
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# ==== CONFIG ====
CONF_THRESH = 0.4
IOU_THRESH = 0.2
RESULTS_DIR = "results"
COIN_INFO = {0: "back", 1: "front"}
last_csv_path = None

if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(base_path, "weights", "best.pt")
model = YOLO(MODEL_PATH)

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# ==== IOU FUNCTION ====
def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    inter_x1, inter_y1 = max(x1, x1g), max(y1, y1g)
    inter_x2, inter_y2 = min(x2, x2g), min(y2, y2g)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    return inter_area / (box1_area + box2_area - inter_area + 1e-6)

# ==== DETECT SINGLE IMAGE ====
def detect_and_display(image_path):
    result_text.set("正在处理 ...请稍等！")
    root.update_idletasks()
    global last_csv_path
    image = cv2.imread(image_path)
    results = model(image)[0]

    filtered_boxes = []
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < CONF_THRESH:
            continue
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        if all(iou(prev["bbox"], xyxy) < IOU_THRESH for prev in filtered_boxes):
            filtered_boxes.append({"cls": int(box.cls[0]), "conf": conf, "bbox": xyxy})

    counts = defaultdict(int)
    detections = []

    for box in filtered_boxes:
        label = COIN_INFO.get(box["cls"], "unknown")
        counts[label] += 1
        detections.append({
            "class": label,
            "confidence": round(box["conf"], 4),
            "bbox": box["bbox"].tolist()
        })

    detections = sorted(detections, key=lambda x: (x["bbox"][1], x["bbox"][0]))
    for i, det in enumerate(detections, 1):
        det["id"] = i
        x1, y1, x2, y2 = det["bbox"]
        center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
        label = det["class"]
        text = f"{i}"
        color = (0, 255, 0) if label == "back" else (0, 0, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale, thickness = 0.6, 2
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_pos = (center_x - text_width // 2, center_y + text_height // 2)
        cv2.putText(image, text, text_pos, font, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(image, text, text_pos, font, font_scale, color, thickness)

    filename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(RESULTS_DIR, f"{filename}_annotated.jpg")
    csv_path = os.path.join(RESULTS_DIR, f"{filename}_result.csv")

    # Save annotated image
    cv2.imwrite(output_path, image)

    # Save CSV
    with open(csv_path, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "class", "confidence", "x1", "y1", "x2", "y2"])
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            writer.writerow([det["id"], det["class"], det["confidence"], x1, y1, x2, y2])

    last_csv_path = csv_path

    # Show image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(image_rgb)
    img_tk = ImageTk.PhotoImage(img_pil.resize((400, int(400 * image.shape[0] / image.shape[1]))))
    result_label.config(image=img_tk)
    result_label.image = img_tk

    # Update result summary
    summary_text = f"正面: {counts['front']}    反面: {counts['back']}    总: {sum(counts.values())}"
    summary_text += f"\n结果保存为: {os.path.abspath(output_path)}"
    result_text.set(summary_text)

    # Display CSV to listbox
    csv_listbox.delete(0, 'end')
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            csv_listbox.insert('end', ', '.join(row))

# ==== DETECT VIDEO ====
def detect_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_objects = 0
    best_frame = None
    result_text.set("Processing video... Please wait.")
    root.update_idletasks()
    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)[0]
        count = sum(1 for box in results.boxes if float(box.conf[0]) > CONF_THRESH)
        if count > max_objects:
            max_objects = count
            best_frame = frame.copy()
    cap.release()
    if best_frame is None:
        messagebox.showerror("Error", "No valid frame found in the video.")
        return
    temp_path = os.path.join(RESULTS_DIR, "temp_best_frame.jpg")
    cv2.imwrite(temp_path, best_frame)
    detect_and_display(temp_path)

# ==== FILE CHOOSER ==== 
def choose_file():
    filepath = filedialog.askopenfilename(filetypes=[("Media Files", "*.jpg *.jpeg *.png *.mp4 *.avi")])
    if filepath:
        ext = os.path.splitext(filepath)[1].lower()
        if ext in [".jpg", ".jpeg", ".png"]:
            detect_and_display(filepath)
        elif ext in [".mp4", ".avi"]:
            detect_video(filepath)
        else:
            messagebox.showerror("Unsupported", "Only image or video files are supported.")

# ==== UI SETUP ====
root = Tk()
root.title("Coin Detect App")
root.geometry("900x700")

Label(root, text='人民币硬币识别目标检测系统', fg='blue', font=('Cambria', 18), width=50).grid(row=0, column=0, columnspan=2, pady=10)

result_text = StringVar()
result_summary = Label(root, textvariable=result_text, font=("Cambria", 12), justify="center")
result_summary.grid(row=1, column=0, columnspan=2, pady=5)

result_label = Label(root)
result_label.grid(row=2, column=0, padx=10, pady=10)

csv_listbox = Listbox(root, width=50, height=25)
csv_listbox.grid(row=2, column=1, padx=10, pady=10)

button_frame = Frame(root)
button_frame.grid(row=2, column=2, columnspan=2, pady=20)

btn_choose = Button(button_frame, text="上传照片", command=choose_file, fg='white', bg='green', font=("Cambria", 14))
btn_choose.pack(side='left', padx=10)

btn_quit = Button(button_frame, text="退出", fg='white', bg='red', font=("Cambria", 14), command=root.quit)
btn_quit.pack(side='left', padx=10)

root.mainloop()
