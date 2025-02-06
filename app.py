import os
import streamlit as st
import whisper
from gtts import gTTS
import io
from groq import Groq
from bs4 import BeautifulSoup
import requests
import pdfplumber  # For PDF extraction
import pytesseract  # For OCR if necessary
from PIL import Image  # For converting PDF to images if needed
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Load the GROQ API key from the environment
# api_key = os.getenv("GROQ_API_KEY")
api_key = "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"

# Initialize the Groq client for LLM
client = Groq(api_key=api_key)

# Load the Whisper model
model = whisper.load_model("base")

# Function to scrape the website content and extract text
def scrape_website(url="https://aibytec.com/"):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text(separator="\n", strip=True)  # Extract plain text from the webpage
        return text
    except Exception as e:
        return f"Error scraping website: {e}"

# Function to extract text from a PDF file using pdfplumber
def extract_pdf_text(pdf_path="/content/aibytec_data.pdf"):
    try:
        pdf_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            first_page = pdf.pages[0]  # You can change the page number if needed
            pdf_text = first_page.extract_text()
            if pdf_text:
                return pdf_text
            else:
                return "No text extracted from this page."
    except Exception as e:
        return f"Error reading PDF: {e}"

# Function to apply OCR on the PDF if text extraction fails
def ocr_pdf(pdf_path="/content/aibytec_data.pdf"):
    try:
        from pdf2image import convert_from_path
        pages = convert_from_path(pdf_path, 300)  # Convert PDF to images at 300 DPI

        ocr_text = ""
        for page in pages:
            # Use pytesseract to extract text from the image
            ocr_text += pytesseract.image_to_string(page)
        return ocr_text if ocr_text else "OCR text extraction failed."
    except Exception as e:
        return f"Error applying OCR: {e}"

# Function to get embeddings using Groq API
def get_embeddings(text):
    try:
        embedding_response = client.embeddings.create(
            model="groq-embedding-model",  # Replace with the appropriate model for embeddings
            inputs=[text]
        )
        return embedding_response['data']
    except Exception as e:
        return f"Error generating embeddings: {e}"

# Function to find the most relevant content based on embeddings
def find_relevant_content(transcribed_text, dataset_text):
    relevant_content = ""
    for sentence in dataset_text.split("\n"):
        if any(keyword.lower() in sentence.lower() for keyword in transcribed_text.split()):
            relevant_content += sentence + "\n"

    if not relevant_content:
        relevant_content = "No relevant content found in the dataset."

    return relevant_content

# Function to process audio, generate a response, and convert it to speech
def process_audio(file_path):
    try:
        # Scrape the dataset from the website (https://aibytec.com/)
        dataset_text = scrape_website()

        # Extract content from the PDF file (/content/aibytec_data.pdf)
        pdf_text = extract_pdf_text()
        if pdf_text == "No text extracted from this page.":
            pdf_text = ocr_pdf()  # Apply OCR if no text was found from the PDF

        # Combine the website content and the PDF content
        combined_dataset = dataset_text + "\n" + pdf_text

        # Transcribe the audio using Whisper
        audio = whisper.load_audio(file_path)
        result = model.transcribe(audio)
        transcribed_text = result["text"]

        # Find the most relevant content from the combined dataset using embeddings
        relevant_content = find_relevant_content(transcribed_text, combined_dataset)

        # Generate a response using Groq by including the transcribed text and relevant content from the dataset
        combined_prompt = f"User input: {transcribed_text}\nRelevant dataset content: {relevant_content}"

        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": combined_prompt}],
            model="llama3-70b-8192"
        )

        # Get the response message
        response_message = chat_completion.choices[0].message.content.strip()

        # Convert the response text to speech
        tts = gTTS(response_message)
        response_audio_io = io.BytesIO()
        tts.write_to_fp(response_audio_io)
        response_audio_io.seek(0)

        # Save the response audio to a file
        with open("response.mp3", "wb") as audio_file:
            audio_file.write(response_audio_io.getvalue())

        return response_message, "response.mp3"

    except Exception as e:
        return f"An error occurred: {e}", None

# Streamlit interface
def main():
    st.title("Audio Processing and Response Generation")

    uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])

    if uploaded_file is not None:
        # Save the uploaded audio to a temporary file
        with open("uploaded_audio.wav", "wb") as f:
            f.write(uploaded_file.read())

        # Process the uploaded audio
        response_message, response_audio_path = process_audio("uploaded_audio.wav")

        # Display the response text
        st.write(f"Response: {response_message}")

        # Play the response audio
        if response_audio_path:
            st.audio(response_audio_path)

if _name_ == "_main_":
    main()