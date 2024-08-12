from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import fitz  # PyMuPDF
import pandas as pd
import os
import logging
import spacy
import io
import cv2
import numpy as np
import pytesseract
from google.cloud import vision
from google.cloud.vision_v1 import types
from PIL import Image

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust the path as needed

# Set up Google Cloud Vision API
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"C:\Users\Sneha\Downloads\festive-quanta-431215-v4-75466e3f6f65.json"
client = vision.ImageAnnotatorClient()

logging.info("Google Cloud Vision API client initialized successfully.")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_form_fields_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    fields = {}
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        for field in page.widgets():
            name = field.field_name
            value = field.field_value
            if field.field_type == fitz.PDF_WIDGET_TYPE_CHECKBOX:
                value = field.field_value if field.field_value else "Not selected"
            if name and value:
                fields[name] = value
    doc.close()
    return fields

def form_fields_to_structured_format(fields):
    structured_data = {}
    for key, value in fields.items():
        if key not in structured_data:
            structured_data[key] = value
    return structured_data

def preprocess_image(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_image = cv2.fastNlMeansDenoising(thresh, None, 30, 7, 21)
    return Image.fromarray(processed_image)

def ocr_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    ocr_results = {}
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img = preprocess_image(img)

        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        content = img_byte_arr.getvalue()
        image = types.Image(content=content)
        response = client.document_text_detection(image=image)
        texts = response.text_annotations

        if texts:
            full_text = texts[0].description
            lines = full_text.split('\n')
            for line in lines:
                if line.strip():
                    ocr_results[line] = ocr_results.get(line, "")
    doc.close()
    return ocr_results

def extract_details_from_text(ocr_results):
    details = {}
    for key, value in ocr_results.items():
        if ':' in key:
            field_name, field_value = key.split(':', 1)
            field_name = field_name.strip()
            field_value = field_value.strip()
            details[field_name] = field_value
        else:
            details[key] = value
    return details

def format_data_for_excel(form_fields, ocr_results):
    data = []
    for field_name, value in form_fields.items():
        data.append({"Field Name": field_name, "Answer": value})
    for field_name, value in ocr_results.items():
        data.append({"Field Name": field_name, "Answer": value})
    return pd.DataFrame(data)

def append_to_excel(data, excel_path):
    df_existing = pd.DataFrame()
    try:
        df_existing = pd.read_excel(excel_path)
    except FileNotFoundError:
        pass

    df_combined = pd.concat([df_existing, data], ignore_index=True, sort=False)
    df_combined.to_excel(excel_path, index=False)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files' not in request.files:
        return jsonify({'error': 'No file part'})
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No selected files'})
    
    pdf_type = request.form.get('pdfType')
    excel_option = request.form.get('excel_option')
    append = excel_option == 'append'
    print(f"PDF type received: {pdf_type}")
    print(f"Excel option received: {excel_option}")
    
    all_form_fields = {}
    all_ocr_results = {}
    for file in files:
        pdf_path = os.path.join("uploads", file.filename)
        file.save(pdf_path)

        form_fields = extract_form_fields_from_pdf(pdf_path)
        structured_form_fields = form_fields_to_structured_format(form_fields)
        for key, value in structured_form_fields.items():
            if key in all_form_fields:
                all_form_fields[key].extend(value)
            else:
                all_form_fields[key] = value

        ocr_results = ocr_images_from_pdf(pdf_path)
        details = extract_details_from_text(ocr_results)
        for key, value in details.items():
            if key in all_ocr_results:
                all_ocr_results[key].extend(value)
            else:
                all_ocr_results[key] = value

    formatted_data = format_data_for_excel(all_form_fields, all_ocr_results)
    excel_path = os.path.join("uploads", "output.xlsx")
    append_to_excel(formatted_data, excel_path)

    return send_file(excel_path, as_attachment=True)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
