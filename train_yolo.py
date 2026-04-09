from ultralytics import YOLO

# Modèle pré-entraîné YOLOv8n (nano) que nous allons fine-tuner
model = YOLO("yolov8n.pt")

# Chemin vers le dataset Roboflow que tu as téléchargé
data_yaml = "datasets/buoy_and_boat/data.yaml"

# Entraînement
model.train(data=data_yaml, epochs=50, imgsz=640)