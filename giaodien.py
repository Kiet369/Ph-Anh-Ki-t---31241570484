!pip install -q gradio #Cài đặt thư viện Gradio

!pip install -q ultralytics #cài đặt thư viện ultralytics

!pip install qrcode[pil] fpdf

!pip install -q ultralytics gradio tensorflow opencv-python pillow qrcode[pil] fpdf pandas

import gradio as gr
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from PIL import Image
import os
import qrcode
import csv
import pandas as pd
from datetime import datetime

# Load models
yolo_model = YOLO("/content/drive/MyDrive/yolov8n.pt")
keras_model_path = "/content/drive/MyDrive/food_model.keras"
tflite_model_path = "/content/drive/MyDrive/food_model.tflite"
if not os.path.exists(tflite_model_path):
    print("Đang convert model .keras sang .tflite...")
    model = tf.keras.models.load_model(keras_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
    print("Convert thành công")

interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Dữ liệu
class_names = [
    "Ca hu kho", "Canh cai", "Canh chua", "Com trang", "Dau hu sot ca",
    "Ga chien", "Rau muong xao", "Thit kho", "Thit kho trung", "Trung chien"
]

food_prices = {
    "Ca hu kho": 22000, "Canh cai": 9000, "Canh chua": 10000,
    "Com trang": 5000, "Dau hu sot ca": 16000, "Ga chien": 25000,
    "Rau muong xao": 8000, "Thit kho": 17000, "Thit kho trung": 18000,
    "Trung chien": 12000
}

nutrition_info = {
    "Ca hu kho": {"calo": 250, "protein": 20, "fat": 18, "carb": 5},
    "Canh cai": {"calo": 40, "protein": 2, "fat": 1, "carb": 5},
    "Canh chua": {"calo": 50, "protein": 3, "fat": 1, "carb": 6},
    "Com trang": {"calo": 200, "protein": 4, "fat": 0.5, "carb": 45},
    "Dau hu sot ca": {"calo": 180, "protein": 10, "fat": 8, "carb": 12},
    "Ga chien": {"calo": 300, "protein": 25, "fat": 20, "carb": 10},
    "Rau muong xao": {"calo": 80, "protein": 2, "fat": 6, "carb": 4},
    "Thit kho": {"calo": 280, "protein": 22, "fat": 21, "carb": 7},
    "Thit kho trung": {"calo": 320, "protein": 25, "fat": 24, "carb": 6},
    "Trung chien": {"calo": 180, "protein": 12, "fat": 15, "carb": 1}
}

food_translations = {
    "Ca hu kho": "Braised Mackerel",
    "Canh cai": "Cabbage Soup",
    "Canh chua": "Sour Soup",
    "Com trang": "White Rice",
    "Dau hu sot ca": "Tofu with Fish Sauce",
    "Ga chien": "Fried Chicken",
    "Rau muong xao": "Stir-fried Water Spinach",
    "Thit kho": "Braised Pork",
    "Thit kho trung": "Braised Pork with Egg",
    "Trung chien": "Fried Egg"
}

def save_history(foods, total, timestamp):
    with open("history.csv", "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, foods.replace("\n", "; "), total])

def load_history():
    if not os.path.exists("history.csv"):
        return "Chưa có giao dịch nào."
    df = pd.read_csv("history.csv", names=["Thời gian", "Danh sách món", "Tổng tiền"])
    return df.to_markdown(index=False)

def classify_image(image):
    results = yolo_model(image)
    detections = results[0].boxes.data.cpu().numpy()

    predicted_classes = []
    cropped_images = []
    total_price = 0
    total_nutrition = {"calo": 0, "protein": 0, "fat": 0, "carb": 0}

    for det in detections:
        x1, y1, x2, y2, score, _ = det
        if score < 0.4:
            continue
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        resized = cv2.resize(crop, (224, 224))
        input_data = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_index = int(np.argmax(output_data))
        confidence = float(np.max(output_data))
        if confidence < 0.5:
            continue
        label_vi = class_names[predicted_index]
        label_en = food_translations.get(label_vi, label_vi)
        if label_en not in predicted_classes:
            total_price += food_prices[label_vi]
            nutri = nutrition_info[label_vi]
            for key in total_nutrition:
                total_nutrition[key] += nutri[key]
        predicted_classes.append(label_en)
        cropped_images.append(crop)

    if not predicted_classes:
        return "Không phát hiện được món ăn", "0₫", "", None, []

    unique_foods = list(dict.fromkeys(predicted_classes))
    result_text = "\n".join([
        f"{en} / {vi}: {food_prices[vi]:,}₫"
        for en, vi in [
            (food, list(food_translations.keys())[list(food_translations.values()).index(food)])
            for food in unique_foods
        ]
    ])

    nutrition_text = (
        f"Calories / Calo: {total_nutrition['calo']} kcal\n"
        f"Protein: {total_nutrition['protein']} g\n"
        f"Fat / Chất béo: {total_nutrition['fat']} g\n"
        f"Carbs / Tinh bột: {total_nutrition['carb']} g"
    )

    qr_path = None
    try:
        momo_url = f"https://momosv3.apimienphi.com/api/QRCode?phone=0000000000&amount={total_price}&note=ThanhToan"
        img = qrcode.make(momo_url)
        qr_path = "/content/momo_qr.png"
        img.save(qr_path)
    except:
        pass

    save_history(result_text, f"{total_price:,}₫", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    return result_text, f"{total_price:,}₫", nutrition_text, qr_path, cropped_images

custom_css = """
body {
    font-family: 'Segoe UI', sans-serif;
    background: linear-gradient(135deg, #fbe9ff, #f3e5f5);
    color: #4a148c;
}
.gr-button {
    background: linear-gradient(90deg, #9c27b0, #7b1fa2);
    color: white !important;
    font-weight: bold;
    border-radius: 10px !important;
    padding: 12px 20px;
    border: none;
    box-shadow: 0 5px 10px rgba(156, 39, 176, 0.3);
    transition: all 0.2s ease-in-out;
}
.gr-button:hover {
    background: linear-gradient(90deg, #7b1fa2, #4a0072);
    transform: scale(1.05);
}
.gr-image, .gr-gallery, .gr-textbox {
    border-radius: 14px !important;
    border: 1px solid #d1c4e9;
    box-shadow: 0 4px 10px rgba(74, 21, 140, 0.1);
}
h2 {
    text-align: center;
    font-size: 26px;
    font-weight: bold;
    color: #6a1b9a;
    margin-top: 16px;
}
label {
    font-weight: 600 !important;
    color: #4a148c !important;
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("<h2 style='color:#6a1b9a'>Nhận Diện Món Ăn & Giá Tiền</h2>")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="numpy", label="Tải ảnh khay cơm", width=300)
            btn = gr.Button("Nhận diện món ăn")

    with gr.Row():
        with gr.Column():
            food_output = gr.Textbox(label="Món & Giá (VI / EN)", lines=6, interactive=False)
        with gr.Column():
            total_output = gr.Textbox(label="Tổng tiền", lines=6, interactive=False)
        with gr.Column():
            nutrition_output = gr.Textbox(label="Dinh dưỡng (VI / EN)", lines=6, interactive=False)

    with gr.Tab("QR Thanh toán Momo"):
        qr_image = gr.Image(label="Quét QR để thanh toán", height=200)

    with gr.Tab("Các món đã cắt"):
        cropped_output = gr.Gallery(label="Các món đã cắt", columns=3, height=230)

    with gr.Tab("Lịch sử giao dịch"):
        history_text = gr.Textbox(label="Lịch sử đơn hàng", lines=15, interactive=False)
        reload_btn = gr.Button("Tải lại lịch sử")

    btn.click(fn=classify_image,
              inputs=image_input,
              outputs=[food_output, total_output, nutrition_output, qr_image, cropped_output])

    reload_btn.click(fn=load_history, outputs=history_text)

demo.launch()
