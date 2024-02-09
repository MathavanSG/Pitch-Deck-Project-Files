import streamlit as st
import os
import pytesseract
import re

from dotenv import load_dotenv
import PyPDF2
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings,HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from htmlTemplate import css,bot_template,user_template

from langchain.chains.summarize import load_summarize_chain
from transformers import pipeline

import streamlit as st
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from bs4 import BeautifulSoup
import requests
import json


def summarize_text(raw_text):
    summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf")
    summary = summarizer(raw_text, max_length=1000, min_length=150, do_sample=False)[0]['summary_text']
    return summary

# Set path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'


def extract_text_from_image(image):
    # Use pytesseract to extract text from the image
    text = pytesseract.image_to_string(image)
    return text

def extract_text_from_pdf_with_images(pdf_bytes):
    # Convert PDF bytes to images
    images = convert_from_bytes(pdf_bytes)

    # Initialize an empty string to store extracted text
    extracted_text = ""

    # Loop through each image and extract text using OCR
    for img in images:
        text = extract_text_from_image(img)
        extracted_text += text + "\n"

    return extracted_text

def get_text_chunks(cleaned_texts_with_images):
    text_splitter=CharacterTextSplitter(
        chunk_size=1000,
        separator='\n',
        chunk_overlap=200,
        length_function=len
    )
    chunks=text_splitter.split_text(cleaned_texts_with_images)
    return chunks

def get_vectorstore(text_chunks):
    embeddings=OpenAIEmbeddings()
    #embeddings=HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore=FAISS.from_texts(texts=text_chunks,embedding=embeddings)
    return vectorstore

def get_conversation(vectorestore):
    llm=ChatOpenAI()
    memory=ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    conversation_chain=ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorestore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response=st.session_state.conversation({'question':user_question})
    st.session_state.chat_history= response['chat_history']

    for i,message in enumerate(st.session_state.chat_history):
        if i % 2==0:
            st.write(user_template.replace('{{MSG}}',message.content),unsafe_allow_html=True)
        else:
            st.write(bot_template.replace('{{MSG}}',message.content),unsafe_allow_html=True)
def is_pdf(file):
    try:
        # Check file extension
        if file.name.endswith('.pdf'):
            return True
        
        # Check file content
        pdf_reader = PyPDF2.PdfFileReader(file)
        if pdf_reader.numPages > 0:
            return True
    except Exception as e:
        pass
    return False

ddg_search = DuckDuckGoSearchAPIWrapper()

RESULTS_PER_QUESTION = 3

def web_search(query: str, num_results: int = RESULTS_PER_QUESTION):
    results = ddg_search.results(query, num_results)
    return [r["link"] for r in results]

def scrape_text(url: str):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            page_text = soup.get_text(separator=" ", strip=True)
            return page_text
        else:
            return f"Failed to retrieve the webpage: Status code {response.status_code}"
    except Exception as e:
        return f"Failed to retrieve the webpage: {e}"



def main():
    st.set_page_config(page_title='Pitch Deck Chat-Bot', page_icon=':snowflake:')
    load_dotenv()
    

    st.write(css,unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation=None
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history=None


    st.header("Pitch Deck Chat-Bot::snowflake:")
    user_question=st.text_input("Ask questions about your file's")

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your PDF documents")
        pdf_files = st.file_uploader("Upload your PDF here and click on Process my file", accept_multiple_files=True)


        if pdf_files is not None:
   
            if st.button("Process my File"):

    
                if pdf_files:
                    with st.spinner("Processing"):
                        for pdf_file in pdf_files:
                            if is_pdf(pdf_file):
                                st.success(f"{pdf_file.name} is a valid PDF file.")

                                pdf_bytes = pdf_file.read()

                                texts_with_images = extract_text_from_pdf_with_images(pdf_bytes)

                                raw_text=texts_with_images
                                #st.write(raw_text)
                                #summary = summarize_text(raw_text)
                                #st.write("Summary:")
                                #st.write(summary)

                                
                                #get text chunk
                                text_chunks=get_text_chunks(raw_text)
                                #st.write(text_chunks)

                                    

                                # create a vector store bruh

                                vectorstore=get_vectorstore(text_chunks)
                                    
                                st.session_state.conversation = get_conversation(vectorstore)
                                st.success("Pdf File processed successfully")
                            else:
                                st.error(f"{pdf_file.name} is not a valid PDF file.")

        






if __name__ == '__main__':
    main()
