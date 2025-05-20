'''
chatbot_core.py: This module defines the core logic of the PDFchatbot.
It builds a RAG pipeline using LangChain and contains the central logic for PDF and model handling. 

Ollama is a tool that allows us to run LLMs locally on our own computer; it offers an API that can access the LangChain directly.
'''

from langchain_community.document_loaders import PyPDFLoader    # Loads and extracts text from PDF files 
from langchain.text_splitter import CharacterTextSplitter       # Splits long text into smaller overlapping chunks  
from langchain_huggingface import HuggingFaceEmbeddings          # Converts text chunks into numerical vectors (embeddings)
from langchain_community.vectorstores import FAISS                   # Stores and searches embeddings using a vector database
from langchain_ollama import ChatOllama           # Connects to a local LLM (e.g. Mistral) via Ollama
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)   
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def build_qa_chain(pdf_path="testing-samples/example.pdf"):
    loader = PyPDFLoader(pdf_path) # Loads the PDF
    documents = loader.load()[:]

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100) # Generates chunks of the document (LMs cannot handle huge text at once, e.g. a whole 50-page PDF)
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # Generates vector embeddings for each chunk
    db = FAISS.from_documents(docs, embeddings) # Stores the chunks in a FAISS vector db for similarity search (FAISS: Faceboon AI Similarity Search = Facebook's vector database)
    retriever = db.as_retriever() 

    llm = ChatOllama(model="tinyllama", temperature=0) 
    
    # Contextualize question
    contextualize_q_system_prompt = (
        """
        Given a chat history and the latest user question, which may reference context in the chat history,
        formulate a standalone question which can be understood without the chat history. Do NOT answer the
        question, just reformualte it if needed and otherwise return it as is.
        """
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Answer question
    qa_system_prompt = (
    "You are a helpful assistant for question-answering tasks. "
    "Only use information from the provided context below to answer the question. "
    "If the context does not contain the answer, say 'I don't know' — do not make anything up.\n\n"
    "Context:\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt), 
        MessagesPlaceholder("chat_history"), ("human", "{input}"),
    ]) 
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt) 
    
    rag_chain = create_retrieval_chain(
        history_aware_retriever, question_answer_chain
    )

    return rag_chain 