from pyzbar import pyzbar
import cv2
import numpy as np
import os
import re
import time
from Model.utils import log

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
            return

        for barcode in barcodes:
            barcodeData = barcode.data.decode("utf-8")
            return barcodeData


    def cap_detection(self, detections) -> bool:
        for det in detections:
            if isinstance(det, str):
                if det.lower() == "cap":
                    return True
        return False
    

  