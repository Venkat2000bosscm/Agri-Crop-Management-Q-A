# 🌾 Agri Crop Management RAG Q&A System

An AI-powered Retrieval-Augmented Generation (RAG) application that allows users to ask agricultural questions from PDF documents and receive intelligent, context-aware answers with source references.

---

# 🚀 Project Overview

This project combines:

* 📄 PDF document processing
* 🧠 Semantic search
* 🔎 Vector similarity retrieval
* 🤖 Large Language Models (LLMs)
* 🌐 Streamlit frontend

The system retrieves relevant agricultural information from uploaded PDF documents and generates accurate answers using AI.

---

# ✨ Features

✅ Upload and process agricultural PDF documents
✅ Semantic search using embeddings
✅ FAISS vector database integration
✅ AI-generated answers using Groq LLM
✅ Source citations from PDF pages
✅ Streamlit web interface
✅ Context-aware question answering
✅ End-to-end RAG pipeline implementation

---

# 🧠 Tech Stack

| Component       | Technology                        |
| --------------- | --------------------------------- |
| Frontend        | Streamlit                         |
| Framework       | LangChain                         |
| Embeddings      | HuggingFace Sentence Transformers |
| Vector Database | FAISS                             |
| LLM             | Groq (Llama 3.1)                  |
| PDF Loader      | PyPDFLoader                       |
| Language        | Python                            |

---

# 📂 Project Structure

```bash
agri-crop-qa/
│
├── documents/                 # Agricultural PDF documents
├── agri_db/                   # FAISS vector database
├── .env                       # API keys
├── config.py                  # Configuration settings
├── main.py                    # Core RAG pipeline
├── app.py                     # Streamlit frontend
├── requirements.txt           # Project dependencies
├── .gitignore                 # Ignore unnecessary files
└── README.md                  # Project documentation
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/agri-crop-qa.git
cd agri-crop-qa
```

---

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux/Mac

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 API Configuration

## Create `.env` File

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get Groq API key from:

[Groq Console](https://console.groq.com/keys?utm_source=chatgpt.com)

---

# 📄 Add Documents

Place agricultural PDF documents inside the `documents/` folder.

Example PDFs:

* Cotton pest management guide
* Crop management manuals
* Agricultural best practices
* Irrigation guides
* Soil management documents

---

# ▶️ Run the Application

## Run Streamlit App

```bash
python -m streamlit run app.py
```

---

# 🖥️ Application Workflow

```text
User Question
      ↓
Document Retrieval
      ↓
Semantic Search using FAISS
      ↓
Relevant Chunk Extraction
      ↓
Prompt Injection
      ↓
Groq LLM Generation
      ↓
Answer + Sources
```

---

# 🧩 RAG Pipeline

## 1️⃣ Document Loading

PDF documents are loaded using:

```python
PyPDFLoader
```

---

## 2️⃣ Text Chunking

Documents are split using:

```python
RecursiveCharacterTextSplitter
```

This improves retrieval quality.

---

## 3️⃣ Embeddings

Text chunks are converted into vector embeddings using:

```python
sentence-transformers/all-MiniLM-L6-v2
```

---

## 4️⃣ Vector Database

Embeddings are stored in:

```python
FAISS
```

This enables semantic similarity search.

---

## 5️⃣ Retrieval

Relevant chunks are retrieved based on user queries.

---

## 6️⃣ Generation

Retrieved context is sent to:

```python
Groq Llama 3.1
```

The model generates context-aware answers.

---

# 📸 Example

## User Question

```text
How do I manage pests in cotton?
```

## AI Answer

```text
Cotton pests can be managed using integrated pest management (IPM) practices such as regular monitoring, identifying pests early, conserving beneficial insects, and applying pesticides only when economic thresholds are exceeded.
```

## Sources

```text
Page 144 from Cotton pest management guide 2022-23.pdf
```

---

# 🌟 Future Improvements

Planned enhancements:

* Multi-PDF upload support
* Conversational memory
* Chat history
* Hybrid search
* Metadata filtering
* Persistent vector database
* Cloud deployment
* Agentic RAG workflows

---

# 🌐 Deployment Options

You can deploy this project using:

* Streamlit Cloud
* HuggingFace Spaces
* Render
* Railway
* Azure App Service

---

# 🛠️ Troubleshooting

## Streamlit Not Found

```bash
pip install streamlit
```

---

## FAISS Installation Error

```bash
pip install faiss-cpu
```

---

## Missing API Key

Ensure `.env` contains:

```env
GROQ_API_KEY=your_key
```

---

## Vector Database Errors

Delete vector DB folder:

```bash
agri_db/
```

Then rerun the application.

---

# 📚 Learning Outcomes

This project demonstrates:

* Retrieval-Augmented Generation (RAG)
* Embedding models
* Vector databases
* Semantic search
* Prompt engineering
* LLM integration
* Streamlit app development
* AI application orchestration

---

# 🤝 Contributing

Contributions are welcome.

Feel free to fork the repository and submit pull requests.

---

# 📜 License

This project is open-source and available under the MIT License.

---

# 👨‍💻 Author

Developed as an AI Engineering and RAG implementation project.
