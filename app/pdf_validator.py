import os
from pypdf import PdfReader

def is_valid_pdf(path):
    if not os.path.exists(path):
        print(f"File not found: {path}\n")
        return False 
    try:
        reader = PdfReader(path)
        if len(reader.pages) == 0:
            print(f"PDF has no readable pages: {path}\n")
            return False 
        return True
    except Exception as e:
        print(f"Invalid or corrupted PDF: {path}. {e}\n")
        return False 
