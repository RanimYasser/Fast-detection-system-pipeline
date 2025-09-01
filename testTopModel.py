import os
from ultralytics import YOLO

# --- config ---
MODEL_PATH = r"C:\Users\RC-co\Desktop\Fast-detection\Yolo-models\best_top.pt"   # change to your model (e.g., "best.pt")
IMAGE_FOLDER = r"C:\Users\RC-co\Desktop\Fast-detection\captures\top"     # folder with your images
OUTPUT_FOLDER = "runs/detect"  # YOLO saves results here by default

def run_yolo_on_folder(model_path, image_folder, output_folder):
    # Load model
    model = YOLO(model_path)

    # Collect all image paths
    image_paths = [
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    # Run prediction
    results = model.predict(source=image_paths, save=True, project=output_folder)

    print(f"Done! Results saved in: {results[0].save_dir}")

if __name__ == "__main__":
    run_yolo_on_folder(MODEL_PATH, IMAGE_FOLDER, OUTPUT_FOLDER)
