import os
import time
from ultralytics import YOLO

# --- config ---
MODEL_PATH = r"C:\Users\RC-co\Desktop\Fast-detection\Yolo-models\best960.pt"   # change to your model (e.g., "best.pt")
IMAGE_FOLDER = r"C:\Users\RC-co\Desktop\Fast-detection\captures\top"           # folder with your images
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
    if not image_paths:
        print(f"No images found in: {image_folder}")
        return

    # Run prediction and measure wall time
    t0 = time.perf_counter()
    results = model.predict(source=image_paths, save=True, project=output_folder, verbose=False)
    wall_secs = time.perf_counter() - t0

    # Print per-image timings from Ultralytics (ms)
    inference_ms_list = []
    print("\nPer-image timings (ms):")
    for img_path, r in zip(image_paths, results):
        sp = getattr(r, "speed", {}) or {}
        pre  = float(sp.get("preprocess", 0.0))
        infer = float(sp.get("inference", 0.0))
        post = float(sp.get("postprocess", 0.0))
        inference_ms_list.append(infer)
        print(f"- {os.path.basename(img_path)} -> preprocess: {pre:.2f}, inference: {infer:.2f}, postprocess: {post:.2f}")

    # Summary
    if inference_ms_list:
        avg_ms = sum(inference_ms_list) / len(inference_ms_list)
        fps = 1000.0 / avg_ms if avg_ms > 0 else float("inf")
        print(f"\nAverage inference: {avg_ms:.2f} ms/image  (~{fps:.2f} FPS)")
    print(f"Total wall time for {len(image_paths)} images: {wall_secs:.3f} s")

    # Where results were saved
    if results:
        print(f"\nDone! Results saved in: {results[0].save_dir}")

if __name__ == "__main__":
    run_yolo_on_folder(MODEL_PATH, IMAGE_FOLDER, OUTPUT_FOLDER)

