import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import Ollama
from sentence_transformers import SentenceTransformer
import os
import boto3
import json

# Import custom CSS and message templates from htmlTemplates.py
from htmlTemplates import css, bot_template, user_template

# Load environment variables
load_dotenv()

def get_pdf_text(pdf_docs):
    """Extracts text from study material PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() if page.extract_text() else ""
    return text


def ocr(files):
    """
    Extracts text from images using AWS Textract and from PDFs using PyPDF2.
    
    Args:
        files (list): List of file objects (images or PDFs).
        
    Returns:
        str: Extracted text from all files.
    """
    textract = boto3.client('textract')
    extracted_text = ""

    for file in files:
        file_name = file.name

        if file_name.endswith('.pdf'):
            # Handle PDFs using PyPDF2
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                extracted_text += page.extract_text() + "\n"

        else:
            # Handle images using AWS Textract
            file_bytes = file.read()
            response = textract.detect_document_text(Document={'Bytes': file_bytes})

            # Extract text from Textract response
            for item in response['Blocks']:
                if item['BlockType'] == 'LINE':
                    extracted_text += item['Text'] + "\n"

    return extracted_text




def get_text_chunks(text):
    """Splits the text into manageable chunks."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    """Creates a FAISS vector store from the text chunks using SentenceTransformer embeddings."""
    # Load the SentenceTransformer model
    embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    # Wrap the embedding model for compatibility with LangChain
    class SentenceTransformerEmbeddings:
        def __init__(self, model):
            self.model = model
        
        def __call__(self, texts):
            # Generate embeddings for a list of texts
            return self.model.encode(texts)

        def embed_documents(self, texts):
            # Generate embeddings for documents
            return self.model.encode(texts)

        def embed_query(self, text):
            # Generate an embedding for a single query
            return self.model.encode([text])[0]

    # Create an instance of the embedding class
    embeddings = SentenceTransformerEmbeddings(embedding_model)

    # Create FAISS vector store using the embeddings
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    """Initializes the conversation chain using Ollama Llama 3.2."""
    # Initialize Ollama Llama 3.2 (running locally)
    llm = Ollama(model="llama3.2")

    # Set up conversation memory
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)

    # Create a conversational retrieval chain using the vector store and LLM
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def fetch_answer(query, conversation_chain):
    """Fetches the correct answer for a given query from the study material."""
    response = conversation_chain({'question': query})
    return response['answer']

def generate_validation_prompt(correct_answer, student_answer):
    """Generates a prompt for LLM to validate the student's response."""
    query = (
        "Validate the student's response with the correct answer and award marks on a scale of 10. "
        "If everything is correct, award 10 marks. If there is a small spelling mistake, award a little less. "
        "If the answer is completely wrong or blank, award 0 marks.\n"
        "Correct Answer: {correct_answer}\n"
        "Student's Answer: {student_answer}\n"
        "Evaluate the response and provide a detailed assessment along with a score out of 10."
    )
    prompt = query.format(correct_answer=correct_answer, student_answer=student_answer)
    return prompt

def validate_answer(user_question, conversation_chain):
    """Validates the student's answer against the correct answer using LLM and assigns a score."""
    # Fetch correct answer from the document using the LLM conversation chain
    correct_answer = fetch_answer(user_question, conversation_chain)
    print("correct answer",correct_answer)
    # Get the student's answer using OCR (from uploaded answer PDFs)
    student_answer = ocr(st.session_state.answer_pdf)

    # Generate prompt to validate student's answer
    prompt = generate_validation_prompt(correct_answer, student_answer)

    # Initialize Ollama Llama 3.2
    llm = Ollama(model="llama3.2")

    # Pass the prompt to Ollama Llama 3.2 for response generation
    response = llm(prompt)

    # Extract the LLM's assessment and score
    return f"Question: {user_question}\nCorrect Answer: {correct_answer}\nStudent's Answer: {student_answer}\nAssessment: {response}"

def handle_userinput(user_question):
    """Handles user input and generates responses using the conversation chain."""
    if st.session_state.conversation is None:
        st.warning("Please upload and process the documents before asking questions.")
        return

    # Generate validation result
    validation_result = validate_answer(user_question, st.session_state.conversation)

    # Display the validation result
    st.write(validation_result)

def main():
    """Main Streamlit app function."""
    # Load page configuration
    st.set_page_config(page_title="AI Exam Evaluator", page_icon=":books:")
    
    # Custom CSS styling
    st.write(css, unsafe_allow_html=True)

    # Initialize session state variables for conversation and chat history
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "answer_pdf" not in st.session_state:
        st.session_state.answer_pdf = None

    # Page header
    st.header("AI Exam Evaluator :books:")

    # User input for questions
    user_question = st.text_input("Upload your documents or answer sheets, and, if desired, provide additional information. The system will process the input and grade the submissions accordingly.")
    if user_question:
        handle_userinput(user_question)

    # Sidebar for uploading and processing PDF documents
    with st.sidebar:
        st.subheader("Your documents")
        
        # Upload the study material PDFs
        pdf_docs = st.file_uploader("Upload your study material PDFs here and click on 'Process'", accept_multiple_files=True)
        
        # Button to process study material PDFs
        if st.button("Process Study Material"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF document.")
            else:
                with st.spinner("Processing study material..."):
                    # Step 1: Extract text from the uploaded PDFs
                    raw_text = get_pdf_text(pdf_docs)

                    # Step 2: Split text into chunks
                    text_chunks = get_text_chunks(raw_text)

                    # Step 3: Create a vector store from text chunks
                    vectorstore = get_vectorstore(text_chunks)

                    # Step 4: Create a conversation chain with the vector store and LLM
                    st.session_state.conversation = get_conversation_chain(vectorstore)

                    # Indicate success
                    st.success("Study material processed successfully! You can now ask questions.")

        # Upload the answer PDFs
        answer_pdfs = st.file_uploader("Upload answer PDFs here for analysis", accept_multiple_files=True)
        
        # Button to process answer PDFs (Optional functionality)
        if st.button("Process Answers"):
            if not answer_pdfs:
                st.warning("Please upload at least one answer PDF.")
            else:
                with st.spinner("Processing answers..."):
                    # Extract text from the answer PDFs using OCR and store in session state
                    st.session_state.answer_pdf = answer_pdfs
                    st.success("Answer PDFs processed successfully!")

if __name__ == '__main__':
    main()
