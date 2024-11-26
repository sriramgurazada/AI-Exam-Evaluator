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


from htmlTemplates import css, bot_template, user_template


load_dotenv()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() if page.extract_text() else ""
    return text

def ocr(pdf_docs):
    textract = boto3.client('textract')
    extracted_text = ""

    for pdf in pdf_docs:
        image_bytes = pdf.read()

        response = textract.detect_document_text(Document={'Bytes': image_bytes})

        for item in response['Blocks']:
            if item['BlockType'] == 'LINE':
                extracted_text += item['Text'] + "\n"

    return " "



def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    class SentenceTransformerEmbeddings:
        def __init__(self, model):
            self.model = model
        
        def __call__(self, texts):
            return self.model.encode(texts)

        def embed_documents(self, texts):
            return self.model.encode(texts)

        def embed_query(self, text):
            return self.model.encode([text])[0]

    embeddings = SentenceTransformerEmbeddings(embedding_model)

    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = Ollama(model="llama3.2")

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def fetch_answer(query, conversation_chain):
    response = conversation_chain({'question': query})
    return response['answer']

def generate_validation_prompt(correct_answer, student_answer):
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
    correct_answer = fetch_answer(user_question, conversation_chain)
    print("correct answer",correct_answer)
    student_answer = ocr(st.session_state.answer_pdf)

    prompt = generate_validation_prompt(correct_answer, student_answer)

    llm = Ollama(model="llama3.2")

    response = llm(prompt)

    return f"Question: {user_question}\nCorrect Answer: {correct_answer}\nStudent's Answer: {student_answer}\nAssessment: {response}"

def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process the documents before asking questions.")
        return

    validation_result = validate_answer(user_question, st.session_state.conversation)

    st.write(validation_result)

def main():
    st.set_page_config(page_title="AI Exam Evaluator", page_icon=":books:")
    
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "answer_pdf" not in st.session_state:
        st.session_state.answer_pdf = None

    st.header("AI Exam Evaluator :books:")

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)


    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your study material PDFs here and click on 'Process'", accept_multiple_files=True)
        
        if st.button("Process Study Material"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF document.")
            else:
                with st.spinner("Processing study material..."):
                    raw_text = get_pdf_text(pdf_docs)

                    text_chunks = get_text_chunks(raw_text)

                    vectorstore = get_vectorstore(text_chunks)

                    st.session_state.conversation = get_conversation_chain(vectorstore)

                    st.success("Study material processed successfully! You can now ask questions.")


        answer_pdfs = st.file_uploader("Upload answer PDFs here for analysis", accept_multiple_files=True)
        
        if st.button("Process Answers"):
            if not answer_pdfs:
                st.warning("Please upload at least one answer PDF.")
            else:
                with st.spinner("Processing answers..."):
                    st.session_state.answer_pdf = answer_pdfs
                    st.success("Answer PDFs processed successfully!")

if __name__ == '__main__':
    main()
