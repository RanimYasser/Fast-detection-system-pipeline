import os
import cv2
import numpy as np
from pyzbar import pyzbar
import csv

class BarcodeReader:
    def __init__(self, input_folder, output_csv, output_images_dir):
        self.input_folder = input_folder
        self.output_csv = output_csv
        self.output_images_dir = output_images_dir
        os.makedirs(self.output_images_dir, exist_ok=True)

    def _annotate(self, image, barcode, text):
        # Prefer polygon if available; otherwise use rect
        if getattr(barcode, "polygon", None):
            pts = np.array([barcode.polygon], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(image, [pts], True, (0, 255, 0), 2)
            # Place label near the first point
            label_x, label_y = pts[0][0][0], max(10, pts[0][0][1] - 10)
        else:
            x, y, w, h = barcode.rect
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label_x, label_y = x, max(10, y - 10)

        # Text background for readability
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 0.6, 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        cv2.rectangle(image, (label_x, label_y - th - 6), (label_x + tw + 6, label_y + 4), (0, 0, 0), -1)
        cv2.putText(image, text, (label_x + 3, label_y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def barcode_detection(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        barcodes = pyzbar.decode(gray)

        results = []
        for barcode in barcodes:
            data = barcode.data.decode("utf-8")
            btype = barcode.type
            results.append((data, btype, barcode))  # keep barcode object for drawing
        return results

    def process_folder(self):
        csv_rows = []

        for filename in os.listdir(self.input_folder):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                continue

            img_path = os.path.join(self.input_folder, filename)
            image = cv2.imread(img_path)
            if image is None:
                print(f"⚠️ Could not read {filename}")
                continue

            detections = self.barcode_detection(image)

            if detections:
                for (data, btype, barcode) in detections:
                    self._annotate(image, barcode, f"{data} ({btype})")
                    csv_rows.append([filename, data, btype])
            else:
                csv_rows.append([filename, "NO BARCODE FOUND", "N/A"])

            # Save annotated image (even if none found, so you can review)
            save_path = os.path.join(self.output_images_dir, filename)
            cv2.imwrite(save_path, image)

        # Save CSV
        with open(self.output_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Image", "BarcodeData", "BarcodeType"])
            writer.writerows(csv_rows)

        print(f"✅ Results saved to {self.output_csv}")
        print(f"🖼️ Annotated images saved in: {self.output_images_dir}")


if __name__ == "__main__":
    folder = r"C:\Users\RC-co\Desktop\Fast-detection\captures\barcode"   # input folder
    output_csv = "barcode_results.csv"
    output_images_dir = r"C:\Users\RC-co\Desktop\Fast-detection\captures\barcode_annotated"

    reader = BarcodeReader(folder, output_csv, output_images_dir)
    reader.process_folder()
