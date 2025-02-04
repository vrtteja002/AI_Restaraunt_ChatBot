import logging
import os
import tempfile
from typing import List, Optional
import pandas as pd
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import BaseRetriever, Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from tenacity import retry, stop_after_attempt, wait_exponential
from streamlit.external.langchain import StreamlitCallbackHandler
from memory import MEMORY
from utils import DocumentLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)

CHUNK_SIZE = 5000
CHUNK_OVERLAP = 200
MODEL_TEMPERATURE = 0.5
MAX_TOKENS_LIMIT = 16384
BATCH_SIZE = 100
REQUEST_TIMEOUT = int(os.getenv('TIMEOUT_SECONDS', 30))

def get_openai_api_key() -> Optional[str]:
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key not found. Add it to .streamlit/secrets.toml")
        LOGGER.error("OpenAI API key not found")
    return api_key

@st.cache_resource
def initialize_llm(api_key: str = None, retry_count: int = 3):
    api_key = api_key or get_openai_api_key()
    if not api_key:
        return None

    try:
        llm = ChatOpenAI(
            model_name="gpt-4o",
            temperature=MODEL_TEMPERATURE,
            streaming=True,
            openai_api_key=api_key,
            max_tokens=MAX_TOKENS_LIMIT,
            request_timeout=REQUEST_TIMEOUT
        )
        LOGGER.info("LLM initialized successfully")
        return llm
    except Exception as e:
        if retry_count > 0:
            LOGGER.warning(f"Retrying LLM initialization. Attempts left: {retry_count}")
            return initialize_llm(api_key, retry_count - 1)
        LOGGER.error(f"Error initializing OpenAI model: {str(e)}")
        st.error(f"Error initializing OpenAI model: {str(e)}")
        return None

def process_documents_in_batches(splits: List[Document], embeddings: OpenAIEmbeddings):
    try:
        vectordb = None
        total_splits = len(splits)
        progress_bar = st.progress(0)
        
        for i in range(0, total_splits, BATCH_SIZE):
            batch = splits[i:min(i + BATCH_SIZE, total_splits)]
            if vectordb is None:
                vectordb = Chroma.from_documents(batch, embeddings)
            else:
                vectordb.add_documents(batch)
            
            progress = min(1.0, (i + BATCH_SIZE) / total_splits)
            progress_bar.progress(progress)
            
        progress_bar.empty()
        return vectordb
    except Exception as e:
        LOGGER.error(f"Error in batch processing: {str(e)}")
        st.error(f"Error processing documents: {str(e)}")
        return None

@st.cache_resource
def configure_retriever(_docs: List[Document]) -> Optional[BaseRetriever]:
    if not _docs:
        st.error("No documents provided for retrieval configuration.")
        return None

    try:
        valid_docs = [doc for doc in _docs if doc.page_content and len(doc.page_content.strip()) > 0]
        if not valid_docs:
            st.error("No valid document content found")
            return None

        LOGGER.info(f"Processing {len(valid_docs)} valid documents")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
            length_function=len
        )

        with st.spinner('Processing documents...'):
            splits = []
            for doc in valid_docs:
                try:
                    doc_splits = text_splitter.split_documents([doc])
                    splits.extend(doc_splits)
                except Exception as e:
                    LOGGER.warning(f"Error splitting document: {str(e)}")
                    continue

        if not splits:
            st.error("No valid splits generated from documents")
            return None

        embeddings = OpenAIEmbeddings(
            openai_api_key=get_openai_api_key(),
            timeout=REQUEST_TIMEOUT,
            max_retries=3
        )

        with st.spinner('Creating embeddings...'):
            vectordb = process_documents_in_batches(splits, embeddings)

        if not vectordb:
            st.error("Failed to create vector database")
            return None

        return vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 10,
                "fetch_k": 50,  # Increased fetch pool for better diversity
                "lambda_mult": 0.7,  # Adjust MMR diversity weight
                "filter": None,  # Optional filter can be added here
                "include_metadata": True,
                "score_threshold": 0.5  # Only return relevant matches
            }
        )
    except Exception as e:
        LOGGER.error(f"Error configuring retriever: {str(e)}")
        st.error(f"Error configuring retriever: {str(e)}")
        return None

def configure_chain(retriever: BaseRetriever, llm: Optional[ChatOpenAI] = None):
    llm = llm or initialize_llm()
    if not llm:
        st.error("Language model initialization failed.")
        return None

    try:
        from langchain.prompts import PromptTemplate
        
        prompt_template = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template="""
        As a friendly and knowledgeable assistant, use the provided context to help the customer with their question about restaurants. 
        
        Keep in mind:
        - Be polite, welcoming, and professional in your language and tone
        - Focus on providing the most relevant information to address the customer's specific question or request
        - For restaurant listings, include the name, location, cuisine type, price range, and any notable features or offerings
        - If the question can't be fully answered by the context, gently let the customer know and offer alternative suggestions or ask for more details
        - Use the chat history to maintain context and avoid repeating information
        
        Context: {context}
        
        Question: {question}
        
        Chat History: {chat_history}
        
        Response:
        """
    )   
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=MEMORY,
            verbose=True,
            max_tokens_limit=MAX_TOKENS_LIMIT,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": prompt_template}
        )
        return chain
    except Exception as e:
        LOGGER.error(f"Error configuring chain: {str(e)}")
        st.error(f"Error configuring chain: {str(e)}")
        return None

def show_chat_interface(chain):
    if not chain:
        st.error("Conversation chain unavailable. Please check API key and reload.")
        return

    try:
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask about restaurants..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                try:
                    with st.spinner("Processing..."):
                        response = chain(
                            {"question": prompt},
                            callbacks=[StreamlitCallbackHandler(message_placeholder)]
                        )
                        
                        if not response or 'answer' not in response:
                            raise ValueError("Invalid response from chain")
                            
                        full_response = response['answer']
                        message_placeholder.markdown(full_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})

                        if 'source_documents' in response:
                            with st.expander("View Sources", expanded=False):
                                sources_seen = set()
                                source_count = 0
                                
                                for doc in response['source_documents']:
                                    # Skip duplicate content
                                    content_hash = hash(doc.page_content)
                                    if content_hash in sources_seen:
                                        continue
                                        
                                    sources_seen.add(content_hash)
                                    source_count += 1
                                    
                                    if source_count > 10:
                                        break
                                        
                                    st.markdown(f"**Source {source_count}:**")
                                    st.markdown(f"```\n{doc.page_content[:300]}...\n```")
                                    
                                    # Show metadata if available
                                    if doc.metadata:
                                        st.markdown("**Metadata:**")
                                        st.json(doc.metadata)
                        
                except TimeoutError:
                    st.error("Request timed out. Please try again.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    LOGGER.error(f"Chat error: {str(e)}")

    except Exception as e:
        st.error("Chat interface error. Please refresh the page.")
        LOGGER.error(f"Interface error: {str(e)}")

def configure_retrieval_chain(
    uploaded_files: List,
    structured_data: Optional[pd.DataFrame] = None,
    external_data: Optional[List[Document]] = None
) -> Optional[ConversationalRetrievalChain]:
    docs = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for file in uploaded_files:
            try:
                temp_filepath = os.path.join(temp_dir, file.name)
                with open(temp_filepath, "wb") as f:
                    f.write(file.getvalue())

                loaded_docs = DocumentLoader.load_document(temp_filepath)
                if loaded_docs:
                    docs.extend(loaded_docs)
                    LOGGER.info(f"Successfully loaded {len(loaded_docs)} documents from {file.name}")
                else:
                    LOGGER.warning(f"No documents loaded from {file.name}")
            except Exception as e:
                LOGGER.error(f"Error loading file {file.name}: {str(e)}")
                st.warning(f"Error loading file {file.name}. Skipping.")
                continue

    if structured_data is not None and not structured_data.empty:
        try:
            for _, row in structured_data.iterrows():
                docs.append(Document(
                    page_content=str(row.to_dict()),
                    metadata={"source": "structured_data"}
                ))
            LOGGER.info(f"Added {len(structured_data)} rows from structured data")
        except Exception as e:
            LOGGER.error(f"Error processing structured data: {str(e)}")

    if external_data:
        docs.extend(external_data)
        LOGGER.info(f"Added {len(external_data)} external documents")

    if not docs:
        st.error("No valid documents to process for retrieval chain.")
        return None

    retriever = configure_retriever(docs)
    if not retriever:
        st.error("Failed to configure retriever.")
        return None

    chain = configure_chain(retriever=retriever)
    if not chain:
        st.error("Failed to configure conversation chain.")
        return None

    LOGGER.info(f"Successfully configured retrieval chain with {len(docs)} documents")
    return chain