import streamlit as st
from main import main, load_vector_store, create_qa_chain, ask_question
from config import DOCUMENTS_DIR, OPENAI_API_KEY

st.set_page_config(page_title="🌾 Agri Crop Q&A", page_icon="🌾")

st.title("🌾 Agri Crop Management Q&A System")
st.markdown("Ask questions about crop management and pest control")

# Initialize session state
if 'qa_chain' not in st.session_state:
    with st.spinner("Loading system..."):
        vectorstore = load_vector_store()
        if vectorstore:
            st.session_state.qa_chain = create_qa_chain(vectorstore)
            st.success("System loaded!")
        else:
            st.error("No database found. Please run main.py first to create the database.")

# Question input
question = st.text_input("Ask your question:", placeholder="e.g., What are common cotton pests?")

if st.button("Get Answer") and question:
    if 'qa_chain' in st.session_state:
        result = ask_question(st.session_state.qa_chain, question)
        st.markdown("### Answer:")
        st.write(result["answer"])
        
        if result["sources"]:
            st.markdown("### Sources:")
            for i, source in enumerate(result["sources"][:3], 1):
                st.write(f"{i}. {source.metadata.get('source', 'Unknown')}")
    else:
        st.error("System not initialized. Please run main.py first.")