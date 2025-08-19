from datetime import date
from Control.ReadExcel import readExcel, findProduct, matchBarcode, product_from_parts

class Box:
    def __init__(self ):
        self.id = None
        self.barcode = None
        self.expire_date = bool
        self.cap = None
        self.brand = None
        self.flavor = None
        self.capacity = None
        self.product_type = None
        self.status = ""
        self.reason=[]
    
    def set_id(self, box_id):
        self.id = box_id

    def set_barcode(self, barcode):
        self.barcode = barcode

    def set_expire_date(self, expire_date):
        self.expire_date = expire_date

    def set_cap(self, cap):
        self.cap = cap

    def set_brand(self, brand):
        self.brand = brand  

    def set_flavor(self, flavor):
        self.flavor = flavor

    def set_capacity(self, capacity):
        self.capacity = capacity

    def set_product_type(self, product_type):
        self.product_type = product_type

    def set_status(self, status):
        self.status = status

    def set_reason(self, reason):
        self.reason = reason

  #getters
    
    def get_cap(self):
        return self.cap
    def get_brand(self):
        return self.brand
    def get_flavor(self):
        return self.flavor
   
    def get_status(self):
        return self.status  

    def get_reason(self):
        return self.reason
    
    def get_barcode(self):
        return self.barcode
    def get_capacity(self):
        return self.capacity
    def has_expire_date(self):
        return self.expire_date

    # Get Info
    def get_box_info(self):
        """Return all box information as a dictionary."""
        return {
            "id": self.id,
            "barcode": self.barcode,
            "expire_date": self.expire_date,
            "cap": self.cap,
            "brand": self.brand ,
            "flavor": self.flavor,
            "capacity": self.capacity,
            "product_type": self.product_type,
            "reason": self.reason,
            "status": self.status

        }
    def evaluate_box(self):
        rejection_reasons = []

        # Barcode-related
        if self.get_barcode() == "inverted":
            rejection_reasons.append("inverted")
        if not self.get_barcode():
            rejection_reasons.append("unreadable barcode")

        # Expiry date check
        if not self.has_expire_date():
            rejection_reasons.append("missing expire date")

        # Cap check
        if self.get_cap() is False and self.get_capacity() == "1L":
            rejection_reasons.append("unsealed")

        # Batch/product validation (only if we have enough product info + barcode)
        if self.get_brand() and self.get_flavor() and self.get_capacity():
            product_str = product_from_parts(
                self.get_brand(),
                self.get_flavor(),
                self.get_capacity()
            )
            if self.get_barcode():
                if not findProduct(product_str):
                    rejection_reasons.append("not in batch")
                elif not matchBarcode(product_str, self.get_barcode()):
                    rejection_reasons.append("invalid barcode")

        # Final decision
        if rejection_reasons:
            self.set_status("rejected")
            self.set_reason(", ".join(rejection_reasons))
        else:
            self.set_status("accepted")
            self.set_reason(None)

        return self.get_status()


    def evaluate_box2(self):
        rejection_reasons=[]
        if self.get_barcode == "inverted":
            rejection_reasons.append("inverted")
        if not self.get_barcode:
            rejection_reasons.append("unreadable barcode")
        if not self.has_expire_date:
            rejection_reasons.append("missing expiredate")
        if self.get_cap is False and self.get_capacity == "1L":
            rejection_reasons.append("unsealed")          
        if rejection_reasons:
            self.set_status("rejected")
            self.set_reason(",".join(rejection_reasons))
        else:
            self.set_status("accepted")
            self.set_reason(None)
        return self.status

    def to_dict(self):
        return {
            "id": self.id,
            "barcode": self.barcode,
            "expire_date": self.expire_date.isoformat() if self.expire_date else None,
            "cap": self.cap,
            "brand": self.brand,
            "flavor": self.flavor,
            "capacity": self.capacity,
            "product_type": self.product_type,
            "reason": self.reason,
            "status": self.status
        }