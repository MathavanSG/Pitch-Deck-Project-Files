import streamlit as st
import os
import pytesseract
import re
from dotenv import load_dotenv
import PyPDF2
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplate import css, bot_template, user_template
from langchain.chains.summarize import load_summarize_chain
from transformers import pipeline



from langchain.schema.output_parser import StrOutputParser
import requests
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from bs4 import BeautifulSoup
import requests
from langchain.utilities import DuckDuckGoSearchAPIWrapper
import json

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4

import io


from io import BytesIO

import time
class RateLimitError(Exception):
    pass

load_dotenv()

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


    
RESULTS_PER_QUESTION = 3

ddg_search = DuckDuckGoSearchAPIWrapper()


def web_search(query: str, num_results: int = RESULTS_PER_QUESTION):
    results = ddg_search.results(query, num_results)
    return [r["link"] for r in results]


SUMMARY_TEMPLATE = """{text} 
-----------
Using the above text, answer in short the following question: 
> {question}
-----------
if the question cannot be answered using the text, imply summarize the text. Include all factual information, numbers, stats etc if available."""  # noqa: E501
SUMMARY_PROMPT = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)


def scrape_text(url: str):
    # Send a GET request to the webpage
    try:
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the content of the request with BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract all text from the webpage
            page_text = soup.get_text(separator=" ", strip=True)

            # Print the extracted text
            return page_text
        else:
            return f"Failed to retrieve the webpage: Status code {response.status_code}"
    except Exception as e:
        print(e)
        return f"Failed to retrieve the webpage: {e}"


url = "https://blog.langchain.dev/announcing-langsmith/"

scrape_and_summarize_chain = RunnablePassthrough.assign(
    summary = RunnablePassthrough.assign(
    text=lambda x: scrape_text(x["url"])[:10000]
) | SUMMARY_PROMPT | ChatOpenAI(model="gpt-3.5-turbo-1106") | StrOutputParser()
) | (lambda x: f"URL: {x['url']}\n\nSUMMARY: {x['summary']}")

web_search_chain = RunnablePassthrough.assign(
    urls = lambda x: web_search(x["question"])
) | (lambda x: [{"question": x["question"], "url": u} for u in x["urls"]]) | scrape_and_summarize_chain.map()

## This is for Arxiv

# from langchain.retrievers import ArxivRetriever
# 
# retriever = ArxivRetriever()
# SUMMARY_TEMPLATE = """{doc} 
# 
# -----------
# 
# Using the above text, answer in short the following question: 
# 
# > {question}
# 
# -----------
# if the question cannot be answered using the text, imply summarize the text. Include all factual information, numbers, stats etc if available."""  # noqa: E501
# SUMMARY_PROMPT = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)
# 
# 
# scrape_and_summarize_chain = RunnablePassthrough.assign(
#     summary =  SUMMARY_PROMPT | ChatOpenAI(model="gpt-3.5-turbo-1106") | StrOutputParser()
# ) | (lambda x: f"Title: {x['doc'].metadata['Title']}\n\nSUMMARY: {x['summary']}")
# 
# web_search_chain = RunnablePassthrough.assign(
#     docs = lambda x: retriever.get_summaries_as_docs(x["question"])
# )| (lambda x: [{"question": x["question"], "doc": u} for u in x["docs"]]) | scrape_and_summarize_chain.map()



SEARCH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            "Write 3 google search queries to search online that form an "
            "objective opinion from the following: {question}\n"
            "You must respond with a list of strings in the following format: "
            '["query 1", "query 2", "query 3"].',
        ),
    ]
)

search_question_chain = SEARCH_PROMPT | ChatOpenAI(temperature=0) | StrOutputParser() | json.loads

full_research_chain = search_question_chain | (lambda x: [{"question": q} for q in x]) | web_search_chain.map()

WRITER_SYSTEM_PROMPT = "You are an AI critical thinker research assistant. Your sole purpose is to write well written, critically acclaimed, objective and structured reports on given text."  # noqa: E501


# Report prompts from https://github.com/assafelovic/gpt-researcher/blob/master/gpt_researcher/master/prompts.py
RESEARCH_REPORT_TEMPLATE = """Information:
--------
{research_summary}
--------
Using the above information, answer the following question or topic: "{question}" in a detailed report -- \
The report should focus on the answer to the question, should be well structured, informative, \
in depth, with facts and numbers if available and a minimum of 1,200 words.
You should strive to write the report as long as you can using all relevant and necessary information provided.
You must write the report with markdown syntax.
You MUST determine your own concrete and valid opinion based on the given information. Do NOT deter to general and meaningless conclusions.
Write all used source urls at the end of the report, and make sure to not add duplicated sources, but only one reference for each.
You must write the report in apa format.
Please do your best, this is very important to my career."""  # noqa: E501

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", WRITER_SYSTEM_PROMPT),
        ("user", RESEARCH_REPORT_TEMPLATE),
    ]
)

def collapse_list_of_lists(list_of_lists):
    content = []
    for l in list_of_lists:
        content.append("\n\n".join(l))
    return "\n\n".join(content)

chain = RunnablePassthrough.assign(
    research_summary= full_research_chain | collapse_list_of_lists
) | prompt | ChatOpenAI(model="gpt-3.5-turbo-1106") | StrOutputParser()


# Function to generate PDF report using ReportLab
# Function to generate PDF report using ReportLab
from reportlab.lib.styles import getSampleStyleSheet

styles = getSampleStyleSheet()
def generate_pdf_report(report_output):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    content = []

    for line in report_output.split("\n"):
        para = Paragraph(line, style=styles["Normal"])
        content.append(para)

    doc.build(content)
    
    # Reset buffer position to start
    buffer.seek(0)
    
    return buffer.getvalue()

def main():
    st.set_page_config(page_title='Integrated Chatbot and Research Assistant', page_icon=':snowflake:')

    st.write(css, unsafe_allow_html=True)

    st.header("Integrated Chatbot and Research Assistant")
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None

    # Initialize chat history attribute in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # User input option
    option = st.sidebar.selectbox("Choose an option", ["PDF Upload", "Research Assistant"])

    if option == "PDF Upload":
        st.header("PDF Upload")
        pdf_files = st.file_uploader("Upload your PDF here and click on Process my file", accept_multiple_files=True)

        if pdf_files is not None:
            if st.button("Process my File"):
                if pdf_files:
                    with st.spinner("Processing"):
                        for pdf_file in pdf_files:
                            if pdf_file.name.endswith('.pdf'):
                                st.success(f"{pdf_file.name} is a valid PDF file.")
                                pdf_bytes = pdf_file.read()
                                texts_with_images = extract_text_from_pdf_with_images(pdf_bytes)
                                raw_text = texts_with_images
                                text_chunks = get_text_chunks(raw_text)
                                vectorstore = get_vectorstore(text_chunks)
                                st.session_state.conversation = get_conversation(vectorstore)
                                st.success("Pdf File processed successfully.")
                            else:
                                st.error(f"{pdf_file.name} is not a valid PDF file.")

    elif option == "Research Assistant":
        st.subheader("Research Assistant")
        user_question = st.text_input("Ask a question")
        if st.button("Get Research Report"):
            if user_question:
                report = chain.invoke({"question": user_question})
                st.write(report)
                st.session_state.chat_history.append(report)
                report_output = f"Research report for '{user_question}':\n{report}"
                text_chunks_r = get_text_chunks(report_output)
                vectorstore_report = get_vectorstore(text_chunks_r)
            
               
                pdf_content = generate_pdf_report(report_output)
            
                # Provide download link for the generated PDF report
                if st.download_button(label="Download your Research Report", data=pdf_content, file_name=f"research_report_{user_question}.pdf", mime="application/pdf"):
                    st.success("Happy Researching with CyberSnow")
                st.session_state.conversation = get_conversation(vectorstore_report)
            

    # Chatbot interaction
    st.header("Chatbot Interaction")
    user_question = st.text_input("Ask questions")

    if user_question:
        if st.session_state.conversation is not None:
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history = response['chat_history']

            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.write(user_template.replace('{{MSG}}', message.content), unsafe_allow_html=True)
                else:
                    st.write(bot_template.replace('{{MSG}}', message.content), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
