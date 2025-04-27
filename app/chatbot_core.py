'''
chatbot_core.py: This module defines the core logic of the PDFchatbot.
It builds a RAG pipeline using LangChain and contains the central logic for PDF and model handeling. 

Ollama is a tool that allows us to run LLMs locally on our own computer; it offers an API that can access the LangChain directly.
'''

from langchain_community.document_loaders import PyPDFLoader    # Loads and extracts text from PDF files 
from langchain.text_splitter import CharacterTextSplitter       # Splits long text into smaller overlapping chunks  
from langchain_huggingface import HuggingFaceEmbeddings          # Converts text chunks into numerical vectors (embeddings)
from langchain_community.vectorstores import FAISS                   # Stores and searches embeddings using a vector database
from langchain_ollama import ChatOllama           # Connects to a local LLM (e.g. Mistral) via Ollama
from langchain.chains import ConversationalRetrievalChain       # Combines LLM, retriever, and chat history into a smart Q&A chain

def build_qa_chain(pdf_path="testing-samples/example.pdf"):
    loader = PyPDFLoader(pdf_path) # Loads the PDF
    documents = loader.load()[:]

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100) # Generates chunks of the document (LMs cannot handle huge text at once, e.g. a whole 50-page PDF)
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # Generates vector embeddings for each chunk
    db = FAISS.from_documents(docs, embeddings) # Stores the chunks in a FAISS vector db for similarity search (FAISS: Faceboon AI Similarity Search = Facebook's vector database)
    retriever = db.as_retriever() # Create a retriever to find relevant chunks based on a question

    llm = ChatOllama(model="mistral") # Combines the retriever with mistral
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain # The function 'qa_chain()' returns a ready-to-use question-answering chain