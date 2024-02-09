CyberSnow PitchDeck

Overview
CyberSnow PitchDeck is an integrated application that serves as both a chatbot and a research assistant. 
It allows users to interact with the system by asking questions and generates detailed research reports based on user queries. 

The application is built using Python and leverages various libraries such as Streamlit, Pytesseract, langchain, transformers, BeautifulSoup, 
and ReportLab.

Features
-PDF Upload: Users can upload PDF files, and the application extracts text and images from the uploaded PDFs and allow the user to interact with
the findings.

-Research Assistant: Users can ask questions, and the application generates detailed research reports with a minimum of 1,200 words.
-Chatbot Interaction: The chatbot component allows users to interact by asking questions directly within the application.


Installation
Install Python 3.6 or higher.
Install required Python packages using pip:

pip install streamlit pytesseract python-dotenv PyPDF2 pdf2image langchain transformers beautifulsoup4 reportlab

Ensure Tesseract OCR is installed and properly configured for pytesseract to work.

Usage
open cmd on the necessary path and the run the command-
strreamlit run app.py

Access the application through a web browser.

Choose between the "PDF Upload" and "Research Assistant" options.

For PDF Upload:
Upload PDF files,you can also upload multiple pdf files.
Click on "Process my File" to extract text and images from PDFs.
The findings are processed on to the memory of the chatbot hence the user can interact easily.

For Research Assistant:

Enter a question in the text input field.
Click on "Get Research Report" to generate a detailed research report.
Interact with the chatbot by asking questions in the designated text input field.

Additional Notes
Customize file paths and configurations as needed, especially for Tesseract OCR and other external dependencies.
Ensure proper setup and configuration of external tools and libraries for smooth functioning of the application.
Feel free to modify and extend the application according to your specific requirements and use cases.

Credits
This project utilizes various open-source libraries and tools, without which it would not have been possible.
