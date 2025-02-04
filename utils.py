# utils.py

import logging
import os
import tempfile
import shutil
from typing import List, Optional, Dict
import pandas as pd
import streamlit as st
import chardet
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader
)
from langchain.schema import Document
import plotly.express as px

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)

class CSVProcessor:
    @staticmethod
    def analyze_csv(df: pd.DataFrame) -> Dict:
        try:
            summary = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'columns': list(df.columns),
                'numeric_columns': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
                'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
                'missing_values': df.isnull().sum().to_dict()
            }

            if summary['numeric_columns']:
                summary['numeric_stats'] = {}
                for col in summary['numeric_columns']:
                    summary['numeric_stats'][col] = {
                        'mean': float(df[col].mean()),
                        'median': float(df[col].median()),
                        'std': float(df[col].std()),
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'quartiles': df[col].quantile([0.25, 0.75]).to_dict()
                    }

            summary['categorical_stats'] = {}
            for col in summary['categorical_columns']:
                value_counts = df[col].value_counts()
                summary['categorical_stats'][col] = {
                    'unique_count': len(value_counts),
                    'top_values': value_counts.head(10).to_dict(),
                    'null_count': df[col].isnull().sum()
                }

            return summary
        except Exception as e:
            LOGGER.error(f"Error analyzing CSV: {str(e)}")
            raise

    @staticmethod
    def plot_numeric_distribution(df: pd.DataFrame, column: str):
        try:
            fig = px.histogram(
                df,
                x=column,
                title=f'Distribution of {column}',
                template='plotly_dark',
                nbins=30
            )
            fig.update_layout(
                showlegend=True,
                xaxis_title=column,
                yaxis_title='Count',
                bargap=0.1
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            LOGGER.error(f"Error plotting distribution: {str(e)}")
            st.error(f"Could not create visualization for {column}")

    @staticmethod
    def plot_categorical_distribution(df: pd.DataFrame, column: str):
        try:
            value_counts = df[column].value_counts().head(10)
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f'Top 10 Categories in {column}',
                template='plotly_dark'
            )
            fig.update_layout(
                xaxis_title=column,
                yaxis_title='Count',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            LOGGER.error(f"Error plotting categories: {str(e)}")
            st.error(f"Could not create visualization for {column}")

class DocumentLoader:
    supported_extensions = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.docx': Docx2txtLoader,
        '.csv': CSVLoader
    }

    @staticmethod
    def load_document(file_path: str) -> List[Document]:
        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in DocumentLoader.supported_extensions:
                raise ValueError(f"Unsupported file type: {ext}")

            loader_class = DocumentLoader.supported_extensions[ext]
            LOGGER.info(f"Loading document: {file_path}")
            loader = loader_class(file_path)
            return loader.load()
        except Exception as e:
            LOGGER.error(f"Error loading document {file_path}: {str(e)}")
            raise

def load_csv(files: List) -> Optional[pd.DataFrame]:
    if not files:
        return None

    dfs = []
    summaries = []

    for file in files:
        try:
            if not file.type in ['text/csv', 'application/vnd.ms-excel']:
                st.error(f"Invalid file type for {file.name}. Must be CSV.")
                continue
                
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_path = tmp_file.name

            with open(tmp_path, 'rb') as f:
                raw = f.read()
                result = chardet.detect(raw)
                encoding = result['encoding']

            df = pd.read_csv(tmp_path, encoding=encoding)
            
            if df.empty:
                st.error(f"No data found in {file.name}")
                continue
                
            summary = CSVProcessor.analyze_csv(df)
            summaries.append({
                'filename': file.name,
                'summary': summary
            })
            dfs.append(df)
            
            st.success(f"Successfully processed {file.name}")

        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
            LOGGER.error(f"CSV processing error: {str(e)}")
            continue

        finally:
            if 'tmp_path' in locals():
                os.unlink(tmp_path)

    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        st.session_state['csv_summaries'] = summaries
        return combined_df
    return None

def fetch_external_data() -> List[Document]:
    try:
        st.sidebar.markdown("### External Data Settings")
        location = st.sidebar.text_input(
            "Search Location",
            value="New York, NY",
            help="Enter city or address to search for restaurants"
        )
        radius = st.sidebar.slider(
            "Search Radius (km)",
            min_value=1,
            max_value=50,
            value=5,
            help="Distance to search from location"
        )
        max_results = st.sidebar.slider(
            "Maximum Results",
            min_value=10,
            max_value=100,
            value=50,
            help="Maximum number of restaurants to fetch"
        )

        api = GooglePlacesAPI()
        restaurants = api.search_restaurants(
            location=location,
            radius=radius * 1000,  # Convert to meters
            max_results=max_results
        )
        
        documents = api.convert_to_documents(restaurants)
        LOGGER.info(f"Fetched {len(documents)} restaurant documents")
        
        return documents
    except Exception as e:
        LOGGER.error(f"Error fetching external data: {str(e)}")
        st.error("Failed to fetch external restaurant data. Please check your API key.")
        return []