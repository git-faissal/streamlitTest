import streamlit as st 
from txtai.pipeline import Summary
from PyPDF2 import PdfReader

#importation des bibliotheque de synthese de presse en ligne
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset, load_metric
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import numpy as np
from transformers import pipeline, set_seed
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import requests
from bs4 import BeautifulSoup
import datetime
#import des package
import nltk
from newspaper import Article
nltk.download('punkt')

st.set_page_config(layout="wide")
@st.cache_resource
#Fonction de resume de texte avec txtai
def summary_text(text):
    summary = Summary()
    text = (text)
    result = summary(text)
    return result

#Fonction extraction du texte d'un document pdf
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as f:
        reader = PdfReader(f)
        page = reader.pages[0]
        text = page.extract_text()
    return text

#Fonction de resume de texte avec Pegasus Turned
def get_response(input_text):
    model_name = 'tuner007/pegasus_summarizer'
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
    max_input_length = 1024  # Maximum length of the input text
    desired_summary_length = len(input_text) // 2  # Calculate desired summary length
    batch = tokenizer([input_text], truncation=True, padding='longest', max_length=max_input_length, return_tensors="pt").to(torch_device)
    gen_out = model.generate(
        **batch,
        max_length=desired_summary_length,  # Set the max length for the summary
        num_beams=5,
        num_return_sequences=1,
        temperature=1.5
    )
    output_text = tokenizer.batch_decode(gen_out, skip_special_tokens=True)
    return output_text


choice = st.sidebar.selectbox(
    "Choisir",
    [
       
        "Resumer un Texte",
        "Resumer un document"
    ]
)

if choice == "Resumer un Texte":
    st.subheader("NB: Dans cette zone, vous pourrez tester le modèle pour résumer vos reportages effectués sur le terrain. Ainsi, vous pourrez permettre à vos lecteurs de lire vos résumés et de se renseigner plus rapidement.")
    st.subheader("Resume de texte avec PegasusTurned")
    input_text = st.text_area("Entrez votre texte ici")
    if st.button("Texte Resume"):
         col1, col2 = st.columns([1,1])
         with col1:
             st.markdown("***Votre texte entrez***")
             st.info(input_text)
         with col2:
             #result = get_response(input_text)
             result = summary_text(input_text)
             st.markdown("***Texte Resume***")
             st.success(result)
             
elif choice == "Resumer un document":
    st.subheader("Document resume avec Pegasus")
    input_file = st.file_uploader("Charger le document", type=['pdf'])
    if input_file is not None:
        if st.button("Resume un document"):
            with open("doc_file.pdf", "wb") as f:
                f.write(input_file.getbuffer())
            col1, col2 = st.columns([1,1])
            with col1:
                st.markdown("**Extraction du texte de votre document**")
                extracted_text = extract_text_from_pdf("doc_file.pdf")
                st.info(extracted_text)
            with col2:
                result = extract_text_from_pdf("doc_file.pdf")
                st.markdown("**Document resume**")
                summary_result = summary_text(result)
                st.success(summary_result)
            