import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Document Configuration
DOCUMENTS_DIR = "documents"
VECTOR_DB_DIR = "agri_db"

# Chunking Configuration
CHUNK_SIZE = 1000  # Number of characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks

# Retrieval Configuration
NUM_RETRIEVED_DOCS = 3  # Number of document chunks to retrieve

# LLM Configuration
TEMPERATURE = 0  # Lower temperature = more focused answers
MODEL_NAME = "gpt-3.5-turbo"  # or "gpt-4" for better quality