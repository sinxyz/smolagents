"""
Advanced RAG (Retrieval Augmented Generation) Chatbot application using Streamlit.

Features:
- Supports multiple LLM providers: OpenAI and local Ollama models.
- Allows document upload (PDF, TXT, MD) for RAG.
- Processes documents, creates vector stores (ChromaDB), and retrieves relevant context.
- Streams responses from LLMs for a more interactive experience.
- Handles errors gracefully and provides user feedback.
- Session state management for chat history and RAG components.
"""
import streamlit as st
import ollama
import openai
import tempfile
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.load import dumps, loads # For potential future use with serialization

# --- Configuration ---
# OpenAI API key is expected to be set as an environment variable `OPENAI_API_KEY`.
# Ollama server is expected to be running for local model access.

# --- Ollama Integration ---
def get_ollama_models():
    """
    Connects to the Ollama API and fetches the list of locally available models.

    Returns:
        list: A list of model names (e.g., ['llama3:latest', 'mistral:latest']).
              Returns an empty list if Ollama is running but has no models.

    Raises:
        ConnectionError: If it cannot connect to the Ollama server or an API error occurs.
    """
    try:
        client = ollama.Client() # Assumes default host (http://localhost:11434)
        models_info = client.list()
        # Extract model names, e.g., "llama2:latest"
        model_names = [model['name'] for model in models_info['models']]
        if not model_names:
            # Ollama is running but no models have been pulled/created
            return []
        return model_names
    except Exception as e:
        # Broad exception catch, as errors can range from connection issues to unexpected API responses
        # Log the error for server-side debugging if a logging framework was integrated
        # print(f"Error fetching Ollama models: {e}")
        raise ConnectionError(f"Could not connect to Ollama or fetch models: {e}")

# --- Document Processing ---
def load_document(uploaded_file):
    """
    Loads an uploaded file (PDF, TXT, MD) and extracts its text content.
    Uses a temporary file to reliably pass the file path to Langchain loaders.

    Args:
        uploaded_file: The file object from `st.file_uploader`.

    Returns:
        str: The extracted text content of the document.
             Returns None if loading fails or the file type is unsupported (though
             `st.file_uploader` should prevent unsupported types).
    """
    if not uploaded_file:
        return None

    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    tmp_file_path = "" # Initialize to ensure it's available in finally block

    try:
        # Create a temporary file to save the uploaded content.
        # This is often necessary because many Langchain loaders expect a file path.
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        text_content = ""
        if file_ext == ".pdf":
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load() # Returns a list of Langchain Document objects
            text_content = "\n".join([doc.page_content for doc in documents])
        elif file_ext in [".txt", ".md"]:
            loader = TextLoader(tmp_file_path, encoding='utf-8')
            documents = loader.load()
            if documents: # TextLoader returns a list with one Document
                text_content = documents[0].page_content
        else:
            # This case should ideally not be reached due to `st.file_uploader` type restrictions.
            st.error(f"Unsupported file type: {file_ext}")
            return None
        return text_content
    except Exception as e:
        st.error(f"Error loading document '{uploaded_file.name}': {e}")
        return None
    finally:
        # Clean up the temporary file
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

def get_text_chunks(text):
    """
    Splits a given text into smaller, manageable chunks for processing by LLMs.

    Args:
        text (str): The input text to be chunked.

    Returns:
        list: A list of text chunks (strings). Returns an empty list if input text is empty.
    """
    if not text:
        st.warning("No text content to chunk.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Max size of each chunk
        chunk_overlap=200, # Number of characters to overlap between chunks
        length_function=len # Function to measure chunk length
    )
    chunks = text_splitter.split_text(text)
    if not chunks:
        st.warning("Text content resulted in no chunks.")
        return []
    return chunks

def get_vector_store(text_chunks, selected_model_name):
    """
    Creates a ChromaDB vector store from text chunks using appropriate embeddings.
    It differentiates between OpenAI and Ollama models to select the correct embedding function.

    Args:
        text_chunks (list): A list of text chunks.
        selected_model_name (str): The name of the LLM selected in the UI, used to
                                   determine the embedding model type (OpenAI vs. Ollama).

    Returns:
        langchain_core.vectorstores.VectorStoreRetriever: A retriever object for the created
                                                         vector store, or None if creation fails.
    """
    if not text_chunks:
        st.warning("No text chunks provided to create vector store.")
        return None

    # Determine if the selected model is an OpenAI model based on its prefix.
    # This is a simplification; a more robust approach might involve explicit model type
    # information passed from the UI or a central model configuration.
    openai_model_prefixes = ("gpt-4", "gpt-3.5")
    is_openai_model = selected_model_name.startswith(openai_model_prefixes)

    embeddings = None
    if is_openai_model:
        try:
            if not os.getenv("OPENAI_API_KEY"):
                st.error("OpenAI API key (OPENAI_API_KEY) is not set in environment variables. "
                         "Cannot create embeddings for OpenAI models.")
                return None
            embeddings = OpenAIEmbeddings()
        except Exception as e:
            st.error(f"Error initializing OpenAI embeddings: {e}")
            return None
    else: # Assume Ollama model
        try:
            # IMPORTANT: Using the selected Ollama *chat* model for embeddings.
            # This may not be optimal as chat models are not always specialized for embeddings.
            # Dedicated Ollama embedding models (e.g., 'nomic-embed-text', 'mxbai-embed-large')
            # would be better. The UI currently doesn't differentiate.
            # If the selected Ollama model is not suitable for embeddings, this step might fail
            # or produce suboptimal embeddings.

            # Check if the Ollama model exists and is accessible.
            # This helps provide a more specific error if the model is unsuitable for embeddings.
            try:
                ollama_client_check = ollama.Client()
                ollama_client_check.show(selected_model_name) # Throws ResponseError if model not found
                embeddings = OllamaEmbeddings(model=selected_model_name)
            except ollama.ResponseError as e:
                 st.error(f"Ollama model '{selected_model_name}' not found or not suitable for generating embeddings. Error: {e}")
                 st.info("Please ensure the selected Ollama model supports embeddings, or try a dedicated Ollama embedding model "
                         "(e.g., 'nomic-embed-text', 'mxbai-embed-large') if available and pulled locally.")
                 return None
            except ConnectionError as e: # More specific catch for Ollama connection issues
                st.error(f"Failed to connect to Ollama server for embeddings: {e}")
                return None
        except Exception as e: # Catch-all for other Ollama embedding initialization errors
            st.error(f"Error initializing Ollama embeddings with model '{selected_model_name}': {e}")
            return None

    if embeddings is None:
        # This should ideally be caught by earlier specific error handling.
        st.error("Failed to initialize any embeddings model.")
        return None

    try:
        # Create an in-memory Chroma vector store from the text chunks and embeddings.
        vector_store = Chroma.from_texts(texts=text_chunks, embedding=embeddings)
        return vector_store.as_retriever() # Return a retriever for querying
    except Exception as e:
        st.error(f"Error creating ChromaDB vector store: {e}")
        return None

# --- Chat Logic ---
def format_rag_prompt(context, question):
    """
    Formats a prompt for Retrieval Augmented Generation (RAG) by combining
    retrieved context with the user's question.

    Args:
        context (str): The retrieved context relevant to the question.
        question (str): The user's original question.

    Returns:
        str: A formatted prompt string ready for the LLM.
    """
    template = """Based on the following context, please answer the user's question. If the context does not contain the answer, clearly state that the context does not provide an answer. Do not try to make up information not present in the provided context.

Context:
{retrieved_chunks}

User Question:
{user_question}"""
    prompt_template = PromptTemplate.from_template(template)
    return prompt_template.format(retrieved_chunks=context, user_question=question)

def get_llm_response(user_query, selected_model_name, retriever=None):
    """
    Retrieves and streams a response from the selected LLM (OpenAI or Ollama).
    If a retriever is provided, it performs RAG by fetching context before querying the LLM.

    Args:
        user_query (str): The user's input query.
        selected_model_name (str): The name of the LLM to use.
        retriever (langchain_core.vectorstores.VectorStoreRetriever, optional):
            The retriever for RAG. If None, RAG is skipped.

    Yields:
        str: Chunks of the LLM's response as they are generated (for streaming).
             Can also yield error messages if issues occur.
    """
    openai_model_prefixes = ("gpt-4", "gpt-3.5") # Consistent with get_vector_store
    is_openai_model = selected_model_name.startswith(openai_model_prefixes)

    final_prompt_for_llm = user_query

    if retriever:
        try:
            # 1. Retrieve relevant chunks using the provided query
            # Note: retriever.invoke returns a list of Document objects
            relevant_documents = retriever.invoke(user_query)

            # 2. Format the retrieved documents into a single context string
            context_str = "\n\n---\n\n".join([doc.page_content for doc in relevant_documents])

            if not context_str:
                st.warning("RAG: No relevant context found for the query.")
                # Proceed with the original query without context, or inform user
            else:
                # 3. Construct the RAG prompt using the formatted context and original query
                final_prompt_for_llm = format_rag_prompt(context=context_str, question=user_query)
                # Optional: Display context for debugging
                # st.sidebar.text_area("RAG Context Retrieved:", value=context_str, height=200)
        except Exception as e:
            st.error(f"Error during RAG context retrieval: {e}")
            # Fallback: Proceed with the original query if RAG fails, or yield an error message
            # For now, we let it proceed with the original query (final_prompt_for_llm is still user_query)
            # Alternatively, could yield f"Error during RAG: {e}. Please try a simpler query or without a document. " and return

    # Select and call the appropriate LLM client
    if is_openai_model:
        try:
            if not os.getenv("OPENAI_API_KEY"):
                st.error("OpenAI API key (OPENAI_API_KEY) not set. Cannot query OpenAI models.")
                yield "Error: OpenAI API key not configured. Please set it as an environment variable. "
                return

            client = ChatOpenAI(model_name=selected_model_name, temperature=0.7, streaming=True)
            messages = [{"role": "user", "content": final_prompt_for_llm}]

            for chunk in client.stream(messages):
                if chunk.content: # Ensure content is not None
                    yield chunk.content
        except Exception as e: # Broad catch for OpenAI client errors
            st.error(f"Error with OpenAI model '{selected_model_name}': {e}")
            yield f"Sorry, an error occurred while communicating with OpenAI: {str(e)}. "
    else: # Ollama model
        try:
            ollama_client = ollama.Client()
            # No explicit check for ollama_client.show(selected_model_name) here,
            # as the model selection dropdown should be populated with available models.
            # If a model is selected, it's assumed to exist. Errors during chat will be caught.

            response_stream = ollama_client.chat(
                model=selected_model_name,
                messages=[{'role': 'user', 'content': final_prompt_for_llm}],
                stream=True
            )
            for chunk in response_stream:
                if chunk['message']['content']: # Ensure content is not None
                    yield chunk['message']['content']
        except ollama.ResponseError as e: # Specific error from Ollama API
            st.error(f"Ollama API error for model '{selected_model_name}': {e.status_code} - {e.error}")
            yield f"Error with Ollama model '{selected_model_name}': {e.error}. Please ensure the model is running and accessible. "
        except ConnectionError as e: # Ollama connection error
            st.error(f"Could not connect to Ollama server: {e}")
            yield "Error: Could not connect to Ollama server. Please ensure Ollama is running. "
        except Exception as e: # Other Ollama errors
            st.error(f"Error with Ollama model '{selected_model_name}': {e}")
            yield f"Sorry, an error occurred while communicating with Ollama: {str(e)}. "

# --- Streamlit UI ---
def main():
    """
    Main function to run the Streamlit application.
    Sets up the UI, handles user interactions, and manages the chat flow.
    """
    st.set_page_config(page_title="Advanced RAG Chatbot", layout="wide")
    st.title("✨ Advanced RAG Chatbot ✨")

    # --- Sidebar for Configuration ---
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Model Selection Logic
        openai_models = ["gpt-4o", "gpt-3.5-turbo"] # Predefined OpenAI models
        ollama_display_models = []
        try:
            local_ollama_models = get_ollama_models()
            if not local_ollama_models:
                ollama_display_models = ["No Ollama models found (pull models first)"]
            else:
                ollama_display_models = local_ollama_models
        except ConnectionError as e:
            # Display connection error in the list of models for user awareness
            ollama_display_models = [f"Ollama N/A (Error: {e})"]
            # Optionally, show a more prominent error/warning in the sidebar
            # st.warning(f"Could not connect to Ollama: {e}")
        except Exception as e: # Catch any other unexpected errors from get_ollama_models
            ollama_display_models = [f"Ollama N/A (Error: {e})"]
            # st.error(f"An unexpected error occurred while fetching Ollama models: {e}")

        # Combine OpenAI and Ollama models for the selectbox
        # Handle cases where Ollama models might be error messages
        if ollama_display_models and ("N/A" in ollama_display_models[0] or "No Ollama models found" in ollama_display_models[0]):
            combined_models = openai_models + ollama_display_models # Show error/status in dropdown
        else:
            combined_models = openai_models + ollama_display_models

        selected_model = st.selectbox("Choose a Model:", combined_models)

        st.markdown("---")
        st.subheader("📄 Document for RAG")
        # File Uploader for RAG
        uploaded_file = st.file_uploader(
            "Upload a document (PDF, TXT, MD) to chat with:",
            type=['txt', 'md', 'pdf'],
            key="file_uploader" # Key helps maintain state across reruns
        )

        # Initialize session state variables for RAG components if they don't exist
        if "retriever" not in st.session_state:
            st.session_state.retriever = None
        if "processed_file_name" not in st.session_state:
            st.session_state.processed_file_name = None

        if uploaded_file:
            # Process the file only if it's new or different from the currently processed one
            if st.session_state.processed_file_name != uploaded_file.name:
                with st.spinner(f"⏳ Processing '{uploaded_file.name}'..."):
                    raw_text = load_document(uploaded_file)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        if text_chunks:
                            # Create vector store and retriever, pass selected_model for embedding choice
                            st.session_state.retriever = get_vector_store(text_chunks, selected_model)
                            if st.session_state.retriever:
                                st.session_state.processed_file_name = uploaded_file.name
                                st.success(f"✅ Document '{uploaded_file.name}' processed and ready for RAG!")
                            else:
                                # Error messages are shown by get_vector_store
                                st.error(f"❌ Could not create RAG retriever for '{uploaded_file.name}'. See errors above.")
                                st.session_state.processed_file_name = None # Reset if processing failed
                                st.session_state.retriever = None
                        else:
                            # Warnings/errors shown by get_text_chunks
                            st.warning(f"⚠️ No text chunks extracted from '{uploaded_file.name}'. Cannot set up RAG.")
                            st.session_state.processed_file_name = None
                            st.session_state.retriever = None
                    else:
                        # Error messages are shown by load_document
                        st.error(f"❌ Failed to load document: '{uploaded_file.name}'.")
                        st.session_state.processed_file_name = None
                        st.session_state.retriever = None
            elif st.session_state.retriever:
                # If the same file is still in the uploader, and it's already processed and retriever exists
                st.success(f"✅ Document '{uploaded_file.name}' is already processed and ready for RAG.")
        elif st.session_state.processed_file_name:
            # If no file is currently uploaded, but one was processed before, offer to clear it.
            st.info(f"Currently using '{st.session_state.processed_file_name}' for RAG.")
            if st.button("Clear loaded document context"):
                st.session_state.retriever = None
                st.session_state.processed_file_name = None
                st.rerun()


    # --- Main Chat Interface ---
    # Initialize chat history in session state if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle new chat input
    if prompt := st.chat_input("Ask your question here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant's response
        with st.chat_message("assistant"):
            response_placeholder = st.empty() # For streaming effect
            full_response_content = ""

            # Retrieve current retriever from session state (it might be None if no doc uploaded/processed)
            current_retriever = st.session_state.get("retriever")

            # Call LLM and stream response
            try:
                for chunk in get_llm_response(prompt, selected_model, current_retriever):
                    if chunk: # Ensure chunk is not None or empty before appending
                        full_response_content += chunk
                        response_placeholder.markdown(full_response_content + "▌") # Simulate typing
                response_placeholder.markdown(full_response_content) # Display final response
            except Exception as e: # Catch any unexpected errors from the response generator
                st.error(f"An unexpected error occurred while generating the response: {e}")
                full_response_content = "Sorry, I encountered an unexpected error while trying to respond."
                response_placeholder.markdown(full_response_content)

        # Add assistant's final response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response_content if full_response_content else "No response generated."})

if __name__ == "__main__":
    main()
