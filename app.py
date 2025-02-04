# app.py

import logging
import sys
import os
from typing import List, Tuple
import pandas as pd
import streamlit as st
from streamlit.external.langchain import StreamlitCallbackHandler

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from chat_with_documents import configure_retrieval_chain, show_chat_interface
from utils import DocumentLoader, load_csv, fetch_external_data, CSVProcessor
from memory import MEMORY

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)

# Constants
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB

def validate_file(uploaded_file) -> bool:
    """
    Validate uploaded file size and type.
    """
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error(f"File {uploaded_file.name} is too large. Maximum size is 200MB")
        return False

    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext not in DocumentLoader.supported_extensions:
        st.error(f"Unsupported file type: {ext}")
        return False
    return True

def process_uploads(files: List) -> Tuple[List, List]:
    """
    Process and validate uploaded files.
    """
    valid_files = [f for f in files if validate_file(f)]
    if not valid_files:
        st.error("No valid files uploaded")
        return [], []

    csv_files = [f for f in valid_files if f.name.lower().endswith('.csv')]
    text_files = [f for f in valid_files if not f.name.lower().endswith('.csv')]
    return csv_files, text_files

def show_csv_analysis(df: pd.DataFrame):
    """
    Display CSV analysis in the sidebar.
    """
    # Dataset Overview
    st.sidebar.markdown("### Dataset Overview")
    st.sidebar.write(f"Total Records: {len(df):,}")
    st.sidebar.write(f"Total Features: {len(df.columns):,}")

    # Column Analysis
    st.sidebar.markdown("### Column Analysis")

    for column in df.columns:
        with st.sidebar.expander(f"📊 {column}", expanded=False):
            st.write(f"**Type:** {df[column].dtype}")
            st.write(f"**Unique Values:** {df[column].nunique():,}")
            st.write(f"**Missing Values:** {df[column].isna().sum():,}")

            try:
                if pd.api.types.is_numeric_dtype(df[column]):
                    CSVProcessor.plot_numeric_distribution(df, column)
                else:
                    CSVProcessor.plot_categorical_distribution(df, column)
            except Exception as e:
                LOGGER.error(f"Error analyzing column {column}: {str(e)}")
                st.error(f"Error analyzing column {column}: {str(e)}")

def setup_sidebar():
    """
    Configure the sidebar with file upload and settings.
    """
    with st.sidebar:
        st.title("📁 Data Upload")

        uploaded_files = st.file_uploader(
            "Upload restaurant datasets",
            type=list(DocumentLoader.supported_extensions.keys()),
            accept_multiple_files=True,
            help="Supported formats: CSV, PDF, DOCX, TXT"
        )

        include_external = st.checkbox(
            "Include external data",
            help="Include additional data from external sources"
        )

        if st.button("🗑️ Clear Chat History"):
            MEMORY.clear()
            st.rerun()

    return uploaded_files, include_external

def show_instructions():
    """Display application instructions."""
    with st.expander("ℹ️ Instructions", expanded=False):
        st.markdown("""
        ### How to Use
        1. **Upload Documents & CSV Data**
            - Use the sidebar to upload restaurant-related files
            - Supported formats: CSV, PDF, DOCX, TXT
            - CSV files will be analyzed automatically

        2. **Ask Questions**
            - Query about menus, ingredients, locations, and trends
            - Analyze pricing and popularity patterns
            - Compare different restaurants or cuisines

        3. **Get Insights**
            - View automatically generated visualizations
            - Get detailed analysis of your data
            - Receive AI-powered recommendations

        ### Example Queries
        - "What are the most common cuisine types in the dataset?"
        - "Show me restaurants with prices under $30"
        - "What are the highest rated restaurants?"
        - "Analyze the price distribution across different cuisines"
        """)

def main():
    """Main application function."""
    try:
        # Page configuration
        st.set_page_config(
            page_title="AI-Powered Restaurant Chatbot",
            page_icon="🍽️",
            layout="wide"
        )

        # Header
        st.title("🍽️ AI-Powered Restaurant Chatbot")

        # Show instructions
        show_instructions()

        # Setup sidebar and get uploaded files
        uploaded_files, include_external = setup_sidebar()

        if not uploaded_files:
            st.info("👈 Please upload restaurant datasets to begin.")
            return

        # Process uploaded files
        with st.spinner("Processing uploaded files..."):
            csv_files, text_files = process_uploads(uploaded_files)

            # Load and process data
            structured_data = None
            if csv_files:
                structured_data = load_csv(csv_files)
                if structured_data is not None and not structured_data.empty:
                    show_csv_analysis(structured_data)

            # Get external data if selected
            external_data = fetch_external_data() if include_external else None

            # Configure the conversation chain
            chain = configure_retrieval_chain(
                text_files,
                structured_data,
                external_data
            )

        # Show chat interface
        if chain:
            show_chat_interface(chain)
        else:
            st.error("Failed to initialize the conversation chain. Please check your inputs and try again.")

    except Exception as e:
        LOGGER.error(f"Application error: {str(e)}")
        st.error("An error occurred while processing your request. Please try again.")
        if st.button("Reset Application"):
            st.rerun()

if __name__ == "__main__":
    main()