import io
import PyPDF2
import docx

def extract_text_from_file(file) -> str:
    """Extract text based on file format (txt, pdf, docx)."""
    filename = file.filename.lower()
    text = ""

    if filename.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            t = page.extract_text()
            if t: text += t + " "

    elif filename.endswith('.docx'):
        doc = docx.Document(file)
        text = " ".join([para.text for para in doc.paragraphs])

    else:
        # Default text (txt or unknown)
        text = file.read().decode('utf-8', errors='ignore')

    return text.strip()
