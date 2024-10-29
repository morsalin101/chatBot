import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Backend processing (using your specific file path)
def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

pdf_text = read_pdf("G:\project\chatBot\waiver.pdf")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_text(pdf_text)

# Generate embeddings and set up FAISS vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = FAISS.from_texts(chunks, embedding=embeddings)
vector_store.save_local("faiss_index")

# Set up retriever
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Configure language model
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise.\n\n{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(model, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Streamlit Interface
st.title("DIU Admission Help Bot")

# User input box for questions
user_question = st.text_input("Ask a question about the waiver or admission:")

# If a question is asked, generate the response
if user_question:
    with st.spinner("Processing..."):
        response = rag_chain.invoke({"input": user_question})
        answer = response['answer']
        st.write("**Response:**", answer)
