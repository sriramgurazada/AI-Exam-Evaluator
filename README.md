**AI Exam Evaluator**

AI Exam Evaluator is an interactive web application that uses large language models (LLMs) to evaluate students' answers by comparing them with correct answers from study material PDFs. Built with Streamlit, LangChain, and FAISS, the application extracts text from study materials, validates student responses, and provides feedback scores.

**Features**

Upload Study Material: Users can upload PDFs of study material, which the application processes for question-answer retrieval.
Conversational Retrieval: The app uses Ollama Llama 3.2 and FAISS to retrieve accurate answers from the material.
Answer Validation: By comparing student answers from answer PDFs with correct responses, the app assigns scores out of 10 and provides detailed feedback.
OCR Integration: AWS Textract extracts text from answer PDFs, enabling the application to read handwritten or typed answers.
Custom UI with CSS: Provides a chat-like interface with avatars for an interactive experience.
Installation

**Prerequisites**
Python 3.8+
AWS Credentials: Required for Textract API access
Packages: streamlit, langchain, sentence-transformers, boto3, PyPDF2, and dotenv

**Setup**
Clone the Repository
Install requirements
Configure AWS Credentials
Run the Application:streamlit run app.py

**Usage**

**Upload Study Material:**
Upload PDFs containing study material in the sidebar and click "Process Study Material."

**Ask Questions:**
Enter questions in the input field to retrieve answers from the study material.

**Validate Answers:**
Upload student answer PDFs, then click "Process Answers" to evaluate and receive feedback.


**Project Structure**

app.py: Main application code integrating Streamlit, LangChain, FAISS, and AWS Textract.
htmlTemplates.py: Contains custom CSS and HTML templates for a chat-based UI.
requirements.txt: List of Python dependencies for easy setup.
.env (optional): Stores sensitive environment variables like AWS credentials (add to .gitignore before uploading to GitHub).


**Key Components**

**Text Extraction from PDFs:**
Utilizes PyPDF2 to read and extract text from study material PDFs.
Integrates AWS Textract to OCR text from answer PDFs.
**Text Splitting and Vector Store:**
Splits the extracted text into smaller, manageable chunks.
Generates vector embeddings using SentenceTransformer and stores them in FAISS for quick retrieval.
**Conversational Chain:**
Sets up a retrieval chain using LangChain and Ollama Llama 3.2 for conversational context and accurate answers.
**Answer Validation:**
Compares the student's response to the correct answer.
Generates detailed feedback and scores based on response accuracy.

**Future Improvements**

**Combination of Multiple Vector Embeddings:** Testing is ongoing to improve response accuracy by combining embeddings.
**Enhanced Query Processing:** Plans to implement additional LLM-based query enhancement and processing for more accurate feedback.


