import re
import pandas as pd

df = None  
PRODUCT_COL = "Product" 
BARCODE_COL = "Barcode"  


def _norm(s: str) -> str:
    # trim, lowercase, collapse spaces
    return re.sub(r"\s+", " ", str(s).strip().casefold())

def product_from_parts(brand: str, flavor: str, capacity: str) -> str:
    return _norm(f"{brand} {flavor} {capacity}")

def readExcel(filename): #haga fel setting up t-call this function
    global df
    df = pd.read_excel(filename, dtype=str)  # Read everything as strings
    df['Barcode'] = df['Barcode'].str.strip()  # Remove leading/trailing spaces
    df['Product'] = df['Product'].str.strip().str.casefold()  # Normalize product names


def matchBarcode(product, barcode):
    p = str(product).strip().casefold()
    b = str(barcode).strip()
    if PRODUCT_COL not in df.columns or BARCODE_COL not in df.columns:
        raise KeyError(f"DataFrame must have '{PRODUCT_COL}' and '{BARCODE_COL}' columns")

    mask = (
        df[PRODUCT_COL].eq(p) &
        df[BARCODE_COL].eq(b)
    )
    return bool(mask.any())

def findProduct(product):
    global df
    product = str(product).strip()  # Ensure input is also clean
    result = df.loc[df['Product'] == product]
    if result.empty:
        print(f"Product {product} not found.")
        #implement rejection logic
        return False
    else:
        print(f"Found in excel:\n{result}")
        return True


