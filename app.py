import streamlit as st
from rag_system import DrugRAGSystem
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Medical RAG System",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Header styling */
    .stMarkdown h1 {
        color: #1E3D59;
        font-size: 2.5rem;
        margin-bottom: 1.5rem;
    }
    
    .stMarkdown h3 {
        color: #17A2B8;
        margin-top: 2rem;
    }
    
    /* Input area styling */
    .stTextArea textarea {
        font-size: 16px !important;
        border: 2px solid #E8ECF1;
        border-radius: 10px;
        padding: 15px;
        min-height: 150px;
    }
    
    .stTextArea textarea:focus {
        border-color: #17A2B8;
        box-shadow: 0 0 0 2px rgba(23, 162, 184, 0.2);
    }
    
    /* Button styling */
    .stButton button {
        background-color: #17A2B8;
        color: white;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background-color: #138496;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Answer container styling */
    .answer-container {
        background-color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1.5rem 0;
        border-left: 5px solid #17A2B8;
    }
    
    /* Source container styling */
    .source-container {
        background-color: #F8F9FA;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    
    .source-title {
        color: #1E3D59;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E8ECF1;
    }
    
    .source-item {
        padding: 1rem;
        background-color: white;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 4px solid #17A2B8;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    .source-item-header {
        font-weight: 600;
        color: #1E3D59;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
    }

    .source-item-content {
        color: #4A5568;
        font-size: 0.9rem;
        line-height: 1.5;
        white-space: pre-wrap;
    }

    .source-item-meta {
        margin-top: 0.5rem;
        font-size: 0.8rem;
        color: #718096;
        border-top: 1px solid #E8ECF1;
        padding-top: 0.5rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    
    /* Loading animation */
    .stSpinner > div {
        border-top-color: #17A2B8 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = DrugRAGSystem()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/medical-doctor.png", width=100)
    st.title("Medical RAG")
    
    st.markdown("""
    ### About
    This AI-powered system provides evidence-based information about:
    - 💊 Drug interactions
    - 🏥 Perioperative management
    - 📚 Clinical research findings
    
    ### Data Sources
    - **Drug Database**: Comprehensive drug interaction data
    - **Academic Research**: Peer-reviewed medical papers
    """)
    
    st.markdown("---")


# Main content
st.title("💉 Medical Research Assistant")
st.markdown("Ask questions about drug interactions, clinical research, and medical procedures.")

# Question input
question = st.text_area(
    "Enter your medical question:",
    height=150,
    placeholder="Example: What are the potential drug interactions of turmeric during perioperative management?"
)

# Process button
col1, col2, col3 = st.columns([1,1,1])
with col2:
    process_button = st.button("🔍 Get Answer", type="primary", use_container_width=True)

if process_button:
    if question:
        with st.spinner("🔄 Analyzing medical literature..."):
            try:
                # Get response
                response = st.session_state.rag_system.query(question)
                
                # Split response into answer and sources
                if "=== Kullanılan Kaynaklar ve İlgili Bölümler ===" in response:
                    answer, sources = response.split("=== Kullanılan Kaynaklar ve İlgili Bölümler ===")
                else:
                    answer, sources = response, ""
                
                # Display answer
                st.markdown("### 📋 Answer")
                st.markdown(f'<div class="answer-container">{answer}</div>', unsafe_allow_html=True)
                
                # Display sources
                if sources:
                    with st.expander("📚 View Sources and References", expanded=True):
                        st.markdown('<div class="source-container">', unsafe_allow_html=True)
                        st.markdown('<div class="source-title">Referenced Sources</div>', unsafe_allow_html=True)
                        
                        # Split sources into sections and process each
                        sections = sources.split("\n\n")
                        for i, section in enumerate(sections, 1):
                            if section.strip():
                                # Split section into title and content if possible
                                parts = section.split(":", 1)
                                if len(parts) == 2:
                                    title, content = parts
                                    st.markdown(
                                        f'''<div class="source-item">
                                            <div class="source-item-header">{title.strip()}</div>
                                            <div class="source-item-content">{content.strip()}</div>
                                            <div class="source-item-meta">Source #{i}</div>
                                        </div>''',
                                        unsafe_allow_html=True
                                    )
                                else:
                                    st.markdown(
                                        f'''<div class="source-item">
                                            <div class="source-item-content">{section.strip()}</div>
                                            <div class="source-item-meta">Source #{i}</div>
                                        </div>''',
                                        unsafe_allow_html=True
                                    )
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
    else:
        st.warning("⚠️ Please enter a question.")
