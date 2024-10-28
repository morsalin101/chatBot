from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS



def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

pdf_text = read_pdf("G:\project\chatBot\waiver.pdf")

splitter = RecursiveCharacterTextSplitter( chunk_size=1000,chunk_overlap=200)
chunks = splitter.split_text(pdf_text)


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = FAISS.from_texts( chunks,embedding=embeddings)
print(vector_store)


