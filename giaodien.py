!pip install -q gradio #Cài đặt thư viện Gradio

!pip install -q ultralytics #cài đặt thư viện ultralytics

import gradio as gr
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from PIL import Image

# Load YOLO model
yolo_model = YOLO("/content/drive/MyDrive/yolov8n.pt")

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="/content/drive/MyDrive/food_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

class_names = [
    "Ca hu kho", "Canh cai", "Canh chua", "Com trang", "Dau hu sot ca",
    "Ga chien", "Rau muong xao", "Thit kho", "Thit kho trung", "Trung chien"
]

food_prices = {
    "Ca hu kho": 10000,
    "Canh cai": 9000,
    "Canh chua": 9000,
    "Com trang": 2000,
    "Dau hu sot ca": 10000,
    "Ga chien": 22000,
    "Rau muong xao": 9000,
    "Thit kho": 32000,
    "Thit kho trung": 15000,
    "Trung chien": 5000
}

def classify_image(image):
    original_img = image.copy()
    results = yolo_model(image)
    detections = results[0].boxes.data.cpu().numpy()

    predicted_classes = []
    cropped_images = []
    total_price = 0

    for det in detections:
        x1, y1, x2, y2, score, class_id = det
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

        predicted_label = class_names[predicted_index]
        predicted_classes.append(predicted_label)
        cropped_images.append(crop)
        total_price += food_prices.get(predicted_label, 0)

        cv2.rectangle(original_img, (x1, y1), (x2, y2), (60, 180, 75), 2)
        cv2.putText(original_img, f"{predicted_label}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 180, 75), 2)

    if not predicted_classes:
        return "Không phát hiện được món ăn", "0đ", image, []

    unique_foods = list(dict.fromkeys(predicted_classes))
    result_text = "\n".join([f"{food}: {food_prices[food]:,}đ" for food in unique_foods])
    return result_text, f"{total_price:,}đ", original_img, cropped_images

custom_css = """
body {
    font-family: 'Segoe UI', sans-serif;
    background: linear-gradient(135deg, #eaf4fc, #fdfbfb);
    background-attachment: fixed;
}

.gr-button {
    background: linear-gradient(to right, #5DADE2, #2E86C1);
    color: white;
    font-weight: bold;
    border-radius: 8px;
    transition: 0.3s ease-in-out;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.15);
}

.gr-button:hover {
    transform: scale(1.03);
    background: #1B4F72;
}

.gr-image, .gr-gallery, .gr-textbox {
    border-radius: 10px !important;
    box-shadow: 0 3px 8px rgba(0, 0, 0, 0.1);
}

h2, p {
    text-align: center;
    color: #154360;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    <h2>Hệ thống Nhận diện Món ăn và Tính tiền</h2>
    <p>Nhận diện các món ăn trên khay cơm và tính toán tổng chi phí tự động.</p>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="numpy", label="Tải ảnh khay cơm")
            btn = gr.Button("Nhận diện & Tính giá tiền")
        with gr.Column(scale=1):
            image_output = gr.Image(type="numpy", label="Ảnh đã nhận diện")

    with gr.Tab("Kết quả nhận diện"):
        food_output = gr.Textbox(label="Danh sách món & giá", lines=10, interactive=False)
        total_output = gr.Textbox(label="Tổng chi phí thanh toán", interactive=False)

    with gr.Tab("Các món đã cắt"):
        cropped_output = gr.Gallery(label="Ảnh các món ăn", columns=3, height=230)

    btn.click(classify_image,
              inputs=image_input,
              outputs=[food_output, total_output, image_output, cropped_output])

demo.launch()
