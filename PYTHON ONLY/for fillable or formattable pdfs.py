import fitz  # PyMuPDF
import pandas as pd

def extract_form_fields_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    fields = {}
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        for field in page.widgets():
            name = field.field_name
            value = field.field_value
            if name and value:
                fields[name] = value
    doc.close()
    return fields

def form_fields_to_structured_format(fields):
    structured_data = {}
    for key, value in fields.items():
        if key not in structured_data:
            structured_data[key] = []
        structured_data[key].append(value)
    return structured_data

def append_to_excel(data, excel_path):
    try:
        df_existing = pd.read_excel(excel_path)
    except FileNotFoundError:
        df_existing = pd.DataFrame()
    
    # structured data to a dataframe
    df_new = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))
    df_combined = pd.concat([df_existing, df_new], ignore_index=True, sort=False)
    df_combined.to_excel(excel_path, index=False)

if __name__ == "__main__":
    pdf_path = r"C:\Users\Sneha\Documents\hackathon pair programming\non_fillable_sample.pdf"
    excel_path = "output.xlsx"

    form_fields = extract_form_fields_from_pdf(pdf_path)
    structured_form_fields = form_fields_to_structured_format(form_fields)
    append_to_excel(structured_form_fields, excel_path)
    print(f"PDF form fields appended to {excel_path}")