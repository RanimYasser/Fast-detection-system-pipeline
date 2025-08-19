from ultralytics import YOLO
m = YOLO(r"C:\Users\RC-co\Desktop\Fast-detection\Yolo-models\best.pt")
m.export(format="onnx", opset=18, nms=True)  # opset ≤ 20 works with DML
