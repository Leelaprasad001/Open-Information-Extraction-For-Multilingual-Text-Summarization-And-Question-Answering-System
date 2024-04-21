from src.imports import *;

def extract_text_url(url):
    try:
        r = requests.get(url)
        soup = BeautifulSoup(r.text, "html.parser")
        text = ' '.join(tag.get_text() for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']))
        return text
    except Exception as e:
        return str(e)
    
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_txt(file):
    text = file.getvalue().decode("utf-8")
    return text

def extract_text_from_docx(file):
    text = docx2txt.process(file)
    text = text.replace('\n', '')
    return text

def extract_text_pdf(file_extension, file):
    if file_extension == "pdf":
        text = extract_text_from_pdf(file)
    elif file_extension == "docx":
        text = extract_text_from_docx(file)
    elif file_extension == "txt":
        text = extract_text_from_txt(file)
    return text
   
def predict_NER(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return doc,nlp

def translate_to_english(text):
    try:
        lang = detect(text)
        if lang == 'en':
            return text
        else:
            translator = Translator()
            translated_text = translator.translate(text, dest='en')
            return translated_text.text
    except:
        return text