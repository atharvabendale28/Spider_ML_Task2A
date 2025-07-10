# Importing all libraries
import os
import pdfplumber  # for better PDF parsing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS  # FAISS instead of Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv
import pickle
import re

load_dotenv()
PAPER_DIR = "papers"

def extract_my_content(pdf_path):
    text = ''
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text(layout=True, x_tolerance=2)
                if page_text:
                    cleaned_text = re.sub(r'\s+', ' ', page_text).strip()
                    cleaned_text = re.sub(r'-\s+', '', cleaned_text)
                    text += cleaned_text + '\n'
        return text
    except Exception as e:
        print(f"An error occurred while processing {pdf_path}: {e}")
        return ''

def load_pdfs(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            
            full_text = extract_my_content(pdf_path)

            if full_text:
                documents.append(Document(page_content=full_text, metadata={"source": filename}))
                print(f"Parsed: {filename}")
    return documents

# Split data into chunks
def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
    return splitter.split_documents(documents)

# Use mpnet-base-v2 sentence transformer
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Build FAISS vector store, and save it to memory using piickle
def build_vector_store(chunks):
    embedding_model = get_embedding_model()
    vector_db = FAISS.from_documents(documents=chunks, embedding=embedding_model)
    with open("faiss_store.pkl", "wb") as f:
        pickle.dump(vector_db, f)
    print("FAISS Vector Store built and saved to faiss_store.pkl")
    return vector_db


# Load FAISS vector store from disk
def load_vector_store(path="faiss_store.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)
    
# Retrieve top k chunks
def retrieve_relevant_chunks(query):
    vectordb = load_vector_store()
    results = vectordb.similarity_search_with_score(query, k=5)
    return [doc for doc, _ in results]
# Generation (uses Groq)
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage

def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",  
        temperature=0.7,
        api_key=os.getenv("GROQ_API_KEY")
    )

def generate_answer(query, context_docs):
    llm = get_llm()
    context_text = " ".join(doc.page_content for doc in context_docs) #joins all text of docs into one string

    system_msg = SystemMessage(content=(
        "You are a helpful AI assistant that answers questions based on academic research papers. "
        "Only use the provided context to answer. If you don't know, say 'I don't know based on the documents.'"
    ))
    human_msg = HumanMessage(content=f"Context:\n{context_text}\n\nQuestion: {query}")

    response = llm.invoke([system_msg, human_msg])
    return response.content

if __name__ == "__main__":
    #only run once to form pkl file
    #docs = load_pdfs(PAPER_DIR)
    #chunks = chunk_documents(docs)
    #build_vector_store(chunks)

    
    #asking the question and using groq to generate answrt
    user_query = input("\nEnter your question: ")
    top_chunks = retrieve_relevant_chunks(user_query)
    answer = generate_answer(user_query, top_chunks)
    print("\nAnswer:\n", answer)










