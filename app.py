import os
import streamlit as st

from main import (
    load_documents,
    split_documents,
    create_vector_store,
    load_vector_store,
    create_qa_chain,
    ask_question
)

from config import (
    DOCUMENTS_DIR,
    VECTOR_DB_DIR
)

# ---------------------------------------------------
# Streamlit Page Config
# ---------------------------------------------------

st.set_page_config(
    page_title="🌾 Agri Crop Q&A",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 Agri Crop Management Q&A System")

st.markdown(
    "Ask questions about crop management and pest control."
)

# ---------------------------------------------------
# Initialize System
# ---------------------------------------------------

@st.cache_resource
def initialize_system():

    faiss_file = os.path.join(VECTOR_DB_DIR, "index.faiss")
    pkl_file = os.path.join(VECTOR_DB_DIR, "index.pkl")

    # Load existing DB
    if os.path.exists(faiss_file) and os.path.exists(pkl_file):

        st.success("✅ Existing vector database loaded")

        vectorstore = load_vector_store()

    # Create new DB
    else:

        st.warning("⚠ No database found. Creating new database...")

        documents = load_documents(DOCUMENTS_DIR)

        chunks = split_documents(documents)

        vectorstore = create_vector_store(chunks)

    qa_system = create_qa_chain(vectorstore)

    return qa_system

# Initialize QA system
qa_system = initialize_system()

# ---------------------------------------------------
# Question Input
# ---------------------------------------------------

question = st.text_input(
    "Ask your question:",
    placeholder="e.g., What are common cotton pests?"
)

# ---------------------------------------------------
# Get Answer
# ---------------------------------------------------

if st.button("Get Answer"):

    if question:

        with st.spinner("Searching documents..."):

            result = ask_question(qa_system, question)

        st.markdown("## 🤖 Answer")

        st.write(result["answer"])

        # Sources
        if result["sources"]:

            st.markdown("## 📚 Sources")

            for i, source in enumerate(result["sources"][:3], 1):

                source_name = source.metadata.get(
                    "source",
                    "Unknown"
                )

                page_num = source.metadata.get(
                    "page",
                    "N/A"
                )

                st.write(
                    f"{i}. Page {page_num} from {source_name}"
                )

    else:

        st.warning("Please enter a question.")