from src.imports import *;
from src.predict import *;

def predict_summarization(text):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizerFast.from_pretrained('./src/models/summarization_model')
    model = EncoderDecoderModel.from_pretrained('./src/models/summarization_model').to(device)

    batch_size = 512
    input_batches = [text[i:i+batch_size] for i in range(0, len(text), batch_size)]

    result = ""
    for i, batch in enumerate(input_batches):
        inputs = tokenizer([batch], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        output = model.generate(input_ids, attention_mask=attention_mask)
        result += tokenizer.decode(output[0], skip_special_tokens=True)

    return result
def perform_qa(question, text):
    question_answerer = pipeline("question-answering", model='distilbert-base-cased-distilled-squad')
    result = question_answerer(question=question, context=text)
    return result['answer'],round(result['score'])

def main():
    st.markdown("<h4 style='text-align: center;'>Open Information Extraction For Multilingual Text Summarization And Question Answering System</h4>", unsafe_allow_html=True)
    input_type = st.selectbox("Select Input Type:", ["Text", "URL","Upload File"])
    extracted_text = ""
    qa_flag = False
    if input_type == "Text":
        text = st.text_area("Enter Text:")
        if st.button("Submit"):
            if text:
                with st.spinner("Extracting Text..."):
                    extracted_text = text
        
    elif input_type == "URL":
        url = st.text_input("Enter Website URL:")
        if st.button("Submit"):
            if url:
                with st.spinner("Extracting Text..."):
                    extracted_text = extract_text_url(url)

    elif input_type == "Upload File":
        uploaded_file = st.file_uploader("Upload file", type=["pdf", "docx", "txt"])

        if uploaded_file is not None:
            file_extension = uploaded_file.name.split(".")[-1].lower()
            with st.spinner("Extracting Text..."):
                extracted_text = extract_text_pdf(file_extension, uploaded_file)
    
    if(extracted_text):
        st.markdown("<h4 style='text-align: center;'>Input Text</h4>", unsafe_allow_html=True)
        extracted_text = translate_to_english(extracted_text)
        st.write(extracted_text)

        st.markdown("<h4 style='text-align: center;'>Question & Answering </h4>", unsafe_allow_html=True)
        with st.form(key="question_form"):
            question = st.text_input("Enter Question:")
            submit_button = st.form_submit_button("Submit Question")
            if submit_button:
                with st.spinner("Performing Question & Answering"):
                    answer, score = perform_qa(question, extracted_text)
                    st.write("Answer:", answer)
                    
        st.markdown("<h4 style='text-align: center;'>Named Entity Recognition (NER) </h4>", unsafe_allow_html=True)
        with st.spinner("Performing Named Entity Recognition..."):
            ner_doc, nlp_model = predict_NER(extracted_text)
            spacy_streamlit.visualize_ner(ner_doc, labels=nlp_model.get_pipe('ner').labels)

        st.markdown("<h4 style='text-align: center;'>Summarization </h4>", unsafe_allow_html=True)
        with st.spinner("Performing Summarization..."):
            summarization_text = predict_summarization(extracted_text)
            st.write(summarization_text) 

        

if __name__ == "__main__":
    main()