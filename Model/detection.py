from pyzbar import pyzbar
import cv2
import numpy as np
import os
import re
import time
# from google.cloud import vision

# === Google Cloud Setup ===
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\kmoha\OneDrive\Desktop\Last-fast-detection\Credentials\key.json"
#client = vision.ImageAnnotatorClient()

class Detect:
    def __init__(self):
        pass

    def label_detections(self, class_name):

        parts = class_name.split("_")

        brand = parts[0] if len(parts) > 0 else "Unknown"
        flavor = parts[1] if len(parts) > 1 else "Unknown"
        capacity = parts[2] if len(parts) > 2 else "Unknown"

        milk_keywords = ['fullcream', 'skimmed', 'laban', 'milk']
        is_milk = any(kw in flavor.lower() for kw in milk_keywords)
        product_type = "milk" if is_milk else "juice"

        return [brand, flavor, capacity, product_type]

    
    def barcode_detection(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        barcodes = pyzbar.decode(gray)
        
        if not barcodes:
            print("No barcodes found in the image.")
            return

        for barcode in barcodes:
            (x, y, w, h) = barcode.rect
            pts = np.array([barcode.polygon], np.int32).reshape(-1, 1, 2)
            cv2.polylines(image, [pts], True, (0, 255, 0), 2)
        
            barcodeData = barcode.data.decode("utf-8")
            barcodeType = barcode.type
            return barcodeData


    def cap_detection(self, detections) -> bool:
        for det in detections:
            if isinstance(det, str):
                if det.lower() == "cap":
                    return True
        return False
    

    # === Date Regex Pattern ===
    date_pattern = r'\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{2,4}[./-]\d{1,2}[./-]\d{1,2}'

    # def expiry_check(self, image):
    #     success, encoded_image = cv2.imencode(".jpg", image)
    #     if not success:
    #         print("❌ Failed to encode image.")
    #         return []

    #     image_bytes = encoded_image.tobytes()
    #     vision_image = vision.Image(content=image_bytes)
    #     response = client.text_detection(image=vision_image)
    #     annotations = response.text_annotations

    #     if not annotations:
    #         print("❌ No text detected.")
    #         return []

    #     full_text = annotations[0].description
    #     dates = re.findall(self.date_pattern, full_text)
    #     if dates:
    #         print("📅 Dates Detected:")
    #         for d in dates:
    #             print(f" - {d}")
    #     else:
    #         print("⚠️ No date patterns found.")

    #     return dates