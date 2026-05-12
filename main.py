import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

from transformers import pipeline
from config import DOCUMENTS_DIR
from config import (
    DOCUMENTS_DIR,
    VECTOR_DB_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    NUM_RETRIEVED_DOCS,
    TEMPERATURE,
    MODEL_NAME,
    OPENAI_API_KEY
)

def load_documents(directory):
    """
    Load all PDF documents from the specified directory.
    
    Args:
        directory: Path to directory containing PDF files
    
    Returns:
        List of loaded documents
    """
    documents = []
    
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist!")
        return documents
    
    # Get all PDF files
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in '{directory}'")
        return documents
    
    print(f"Found {len(pdf_files)} PDF file(s)")
    
    # Load each PDF
    for filename in pdf_files:
        file_path = os.path.join(directory, filename)
        print(f"Loading: {filename}")
        try:
            loader = PyPDFLoader(file_path)
            loaded_docs = loader.load()
            documents.extend(loaded_docs)
            print(f"  ✓ Loaded {len(loaded_docs)} pages")
        except Exception as e:
            print(f"  ✗ Error loading {filename}: {str(e)}")
    
    print(f"\nTotal documents loaded: {len(documents)} pages")
    return documents
def split_documents(documents):
    """
    Split documents into smaller chunks for better retrieval.
    
    Args:
        documents: List of document objects
    
    Returns:
        List of document chunks
    """
    print("\nSplitting documents into chunks...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    print(f"Average chunk size: {CHUNK_SIZE} characters")
    
    return chunks
def create_vector_store(chunks):
    """
    Create a vector database from document chunks.
    
    Args:
        chunks: List of document chunks
    
    Returns:
        Vector store object
    """
    print("\nCreating vector database...")
    
    # Check if API key is set
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in .env file!")
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
    
    # Create vector store
    vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)
    
    print(f"✓ Vector database created in '{VECTOR_DB_DIR}'")
    return vectorstore
def load_vector_store():
    return None
    """
    Load existing vector database if it exists.
    
    Returns:
        Vector store object or None if not found
    """
    if os.path.exists(VECTOR_DB_DIR) and os.listdir(VECTOR_DB_DIR):
        print(f"Loading existing vector database from '{VECTOR_DB_DIR}'...")
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=embeddings
        )
        print("✓ Vector database loaded")
        return vectorstore
    return None
def ask_question(qa_system, question):

    print(f"\nQuestion: {question}")
    print("Searching documents...")

    try:
        retriever = qa_system["retriever"]
        generator = qa_system["generator"]

        docs = retriever.get_relevant_documents(question)

        context = "\n".join([doc.page_content[:200] for doc in docs[:2]])

        prompt = f"""
        Context:
        {context}

        Question:
        {question}

        Answer briefly and clearly:
        """

        response = generator(prompt)

        answer = response[0]["generated_text"].strip()

        return {
            "answer": answer,
            "sources": docs
        }

    except Exception as e:

        return {
            "answer": f"Error: {str(e)}",
            "sources": []
        }

def create_qa_chain(vectorstore):

    print("\nSetting up Q&A system...")

    generator = pipeline(
    task="text-generation",
    model="google/flan-t5-base",
    max_new_tokens=100,
    temperature=0.2
    )

    print("✓ Q&A system ready")

    return {
        "retriever": vectorstore.as_retriever(
            search_kwargs={"k": NUM_RETRIEVED_DOCS}
        ),
        "generator": generator
    }    
def main():
    """
    Main function to initialize and run the Q&A system.
    """
    print("=" * 60)
    print("🌾 Agri Crop Management Q&A System")
    print("=" * 60)
    
    # Try to load existing vector store
    vectorstore = load_vector_store()
    
    # If not found, create new one
    if vectorstore is None:
        print("\nNo existing database found. Creating new one...")
        
        # Load documents
        documents = load_documents(DOCUMENTS_DIR)
        
        if not documents:
            print("No documents found! Please add PDF files to the 'documents' folder.")
            return
        
        # Split into chunks
        chunks = split_documents(documents)
        
        # Create vector store
        vectorstore = create_vector_store(chunks)
    
    # Create Q&A chain
    qa_system = create_qa_chain(vectorstore)
    
    # Interactive Q&A loop
    print("\n" + "=" * 60)
    print("System ready! Ask your questions (type 'quit' to exit)")
    print("=" * 60)
    
    while True:
        question = input("\n👤 Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! 👋")
            break
        
        if not question:
            print("Please enter a question.")
            continue
        
        # Get answer
        result = ask_question(qa_system, question)
        
        # Display answer
        print("\n" + "-" * 60)
        print("🤖 Answer:")
        print("-" * 60)
        print(result["answer"])
        
        # Display sources
        if result["sources"]:
            print("\n" + "-" * 60)
            print("📚 Sources:")
            print("-" * 60)
            for i, source in enumerate(result["sources"][:3], 1):
                page = source.metadata.get('page', 'N/A')
                source_name = source.metadata.get('source', 'Unknown')
                print(f"{i}. Page {page} from {os.path.basename(source_name)}")
        
        print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
