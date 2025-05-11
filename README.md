# PDF Chatbot

**PDF Chatbot** is an AI-powered application that allows users to interact with PDF documents using natural language. It uses a Retrieval-Augmented Generation (RAG) pipeline built with [LangChain](https://www.langchain.com/) and integrates local LLMs via [Ollama](https://ollama.com/). 

This project was made for learning purposes and to get familiar with concepts such as chunking, embeddings, vector search and RAG. 

---

## Installation

1. **Clone the Repository**:
```
git clone https://github.com/your-username/pdf-chatbot.git
```
```
cd pdf-chatbot
```

2. **Set Up a Virtual Environment**:
```
python3 -m venv pdfchatbot
```
```
source pdfchatbot/bin/activate # On Mac/Linux
```

3. **Install Dependencies**:
```
pip install -r requirements.txt
```

---

## Usage 
1. **Streamlit Web Interface**
   Run the Streamlit app to interact with the chatbot via a web interface:

   ```
   cd app
   ```
   ```
   streamlit run streamlit_app.py
   ```
   * Upload a PDF file
   * Ask questions about the document
  
---

## Acknowledgments
This project is based on the following article: "[RAG in Action: Build your Own Local PDF Chatbot as a Beginner](https://medium.com/data-science-collective/rag-in-action-build-your-own-local-pdf-chatbot-as-a-beginner-96c2833869ff)". This project expands on the original structure with additional features and improvements.
