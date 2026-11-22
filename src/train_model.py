from ultralytics import YOLO
import torch

model = YOLO("yolo11n.pt")

results = model.train(
    data="dataset/yolo/data.yaml",
    epochs=30,
    imgsz=256,
    batch=8,
    project=r"C:\Users\swamy\Desktop\AI_Industrial_Quality_Inspection\runs\industrial_defect",
    name="yolo11n_neu",
    device="cpu",
    workers=0,
    patience=10,
    plots=True,
    save=True
)

print("\nTraining completed!")