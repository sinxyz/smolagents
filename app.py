# app.py
"""
Streamlit-based chatbot application featuring RAG (Retrieval Augmented Generation)
and response generation via the `smol_dev` library.

Application Purpose:
--------------------
This application serves as an interactive chatbot that can answer user queries.
It leverages the `smol_dev` library for generating responses, which is an
experimental approach for direct chat/text generation as `smol_dev` is primarily
designed for code generation. The integration attempts to guide `smol_dev` to
produce textual answers by prompting it to write its response to a conceptual
file named "answer.txt".

Key Features:
-------------
- Multi-Model Support: Allows selection between OpenAI models (e.g., gpt-4o, gpt-3.5-turbo)
  and locally hosted Ollama models.
- RAG Pipeline:
    - Users can upload documents (PDF, TXT, MD).
    - Documents are processed (loaded, chunked) and a vector store (ChromaDB)
      is created using appropriate embeddings (OpenAI or Ollama).
    - Retrieved context from these documents can be used to inform the AI's response.
- `smol_dev` for Response Generation:
    - The core response generation logic uses `smol_dev`'s `plan`, `specify_file_paths`,
      and `generate_code_sync` functions.
    - This is an **experimental use case** for `smol_dev`. The prompts are engineered
      to make `smol_dev` output a textual answer rather than code.
    - Environment variables (`OPENAI_API_KEY`, `OPENAI_API_BASE`, `OPENAI_MODEL_NAME`)
      are manipulated to direct `smol_dev` to use the selected model (OpenAI or,
      experimentally, an Ollama model via an OpenAI-compatible API endpoint).
- Ollama Integration:
    - Lists available Ollama models.
    - Attempts to use selected Ollama models for both chat response generation
      (via `smol_dev`'s experimental OpenAI API compatibility) and for embeddings in the RAG pipeline.
      Success with `smol_dev` and Ollama is not guaranteed and depends on the Ollama
      server's OpenAI API compatibility and `smol_dev`'s internal handling of API calls.
- Streamlit UI: Provides a user-friendly interface for model selection, file upload,
  and chat interaction.
- Error Handling & Feedback: Includes mechanisms to handle missing libraries (`smol_dev`),
  API key issues, model connection problems, and RAG processing failures, providing
  feedback to the user through the UI.

Experimental Aspects:
---------------------
- Using `smol_dev` for direct textual chat response generation.
- Attempting to make `smol_dev` (which is primarily an OpenAI-focused tool)
  interact with Ollama models by setting `OPENAI_API_BASE` and other relevant
  environment variables to point to an Ollama server's OpenAI-compatible endpoint.
- Using the selected Ollama chat model for generating embeddings in the RAG pipeline;
  dedicated embedding models are generally preferred for Ollama but this app simplifies
  by using the chat model for both for experimental purposes.

Dependencies:
-------------
Requires Streamlit, Langchain components, OpenAI, Ollama, ChromaDB, PyPDF, Tiktoken,
and `smol_dev`. Ensure `requirements.txt` is used for installation.
The `OPENAI_API_KEY` environment variable must be set for OpenAI model usage.
An Ollama server should be running with desired models pulled for Ollama usage.
"""
import streamlit as st
import os
import tempfile

# For RAG pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings # Assuming smol_dev won't handle Ollama embeddings

# Placeholder for smol_dev imports - will be used later
# from smol_dev.prompts import plan, specify_file_paths, generate_code_sync

# --- Environment Setup ---
# Ensure OPENAI_API_KEY is available in the environment for OpenAI models/embeddings.
# For Ollama, ensure the Ollama server is running and models are pulled.

import ollama # Top-level import for use in get_ollama_models_list and OllamaEmbeddings
# Attempt to import smol_dev; handle if not installed.
# This is crucial for the application's core response generation logic.
try:
    from smol_dev.prompts import plan, specify_file_paths, generate_code_sync
    SMOL_DEV_AVAILABLE = True
except ImportError:
    SMOL_DEV_AVAILABLE = False
    # Define dummy functions if smol_dev is not available.
    # This allows the Streamlit app to load and display a warning,
    # rather than crashing outright if smol_dev is missing.
    def plan(prompt):
        st.error("CRITICAL: smol_dev library not found or failed to import. This app requires smol_dev. Please install it.")
        raise NotImplementedError("smol_dev not available")
    def specify_file_paths(prompt, shared_deps):
        st.error("CRITICAL: smol_dev library not found or failed to import. This app requires smol_dev. Please install it.")
        raise NotImplementedError("smol_dev not available")
    def generate_code_sync(prompt, shared_deps, file_path):
        st.error("CRITICAL: smol_dev library not found or failed to import. This app requires smol_dev. Please install it.")
        raise NotImplementedError("smol_dev not available")

# --- Environment Variable Management for smol_dev ---
# Store original OpenAI environment variables at startup.
# This is to ensure that manipulations for smol_dev calls (especially for Ollama)
# do not permanently alter the environment for other parts of the app or session.
ORIGINAL_OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ORIGINAL_OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE")
ORIGINAL_OPENAI_MODEL = os.environ.get("OPENAI_MODEL_NAME") # Or other smol_dev relevant model env vars

def set_smol_dev_env_for_openai(api_key, model_name=None):
    """
    Configures environment variables for smol_dev to use OpenAI.
    It ensures that `OPENAI_API_KEY` is set and, if provided, `OPENAI_MODEL_NAME`.
    Crucially, it resets `OPENAI_API_BASE` to its original state (or unsets it)
    to prevent accidental routing to a local/Ollama endpoint if it was previously set.
    """
    os.environ["OPENAI_API_KEY"] = api_key
    if model_name:
        os.environ["OPENAI_MODEL_NAME"] = model_name # For smol_dev to pick up the specific model

    # Reset OPENAI_API_BASE to default for OpenAI calls
    if ORIGINAL_OPENAI_API_BASE: # If there was an original base (e.g., for Azure OpenAI), restore it
        os.environ["OPENAI_API_BASE"] = ORIGINAL_OPENAI_API_BASE
    elif "OPENAI_API_BASE" in os.environ: # If no original, but it was set (e.g., for Ollama previously), remove it
        del os.environ["OPENAI_API_BASE"]

def set_smol_dev_env_for_ollama(ollama_model_name, ollama_api_base="http://localhost:11434/v1"):
    """
    Configures environment variables to (experimentally) make smol_dev use an Ollama endpoint.
    This function sets `OPENAI_API_KEY` to a dummy value (as some clients require it to be non-empty),
    `OPENAI_API_BASE` to the Ollama server's OpenAI-compatible API endpoint, and
    `OPENAI_MODEL_NAME` to the selected Ollama model.

    **Experimental Note**: Success depends on the Ollama server correctly mimicking the
    OpenAI API and `smol_dev` respecting these environment variables for its API calls.
    """
    os.environ["OPENAI_API_KEY"] = "ollama_is_great" # Dummy key, as Ollama doesn't use it but client might check
    os.environ["OPENAI_API_BASE"] = ollama_api_base
    os.environ["OPENAI_MODEL_NAME"] = ollama_model_name # Pass the specific Ollama model name

def restore_original_env():
    """
    Restores OpenAI-related environment variables to their original state
    that was captured at application startup. This is critical to call after
    each `smol_dev` operation to ensure subsequent operations (or other parts
    of a larger app) are not affected by temporary changes.
    """
    # Restore OPENAI_API_KEY
    if ORIGINAL_OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = ORIGINAL_OPENAI_API_KEY
    elif "OPENAI_API_KEY" in os.environ: # If no original key, but one was set by us (e.g., dummy for Ollama)
        del os.environ["OPENAI_API_KEY"]

    # Restore OPENAI_API_BASE
    if ORIGINAL_OPENAI_API_BASE:
        os.environ["OPENAI_API_BASE"] = ORIGINAL_OPENAI_API_BASE
    elif "OPENAI_API_BASE" in os.environ: # If no original base, but one was set (e.g., for Ollama)
        del os.environ["OPENAI_API_BASE"]

    # Restore OPENAI_MODEL_NAME (or other smol_dev relevant model env var)
    if ORIGINAL_OPENAI_MODEL:
        os.environ["OPENAI_MODEL_NAME"] = ORIGINAL_OPENAI_MODEL
    elif "OPENAI_MODEL_NAME" in os.environ: # If no original model, but one was set
        del os.environ["OPENAI_MODEL_NAME"]

# --- smol_dev based response generation ---
def get_smol_dev_response(user_query: str, rag_context: str = None, selected_model_name: str = None, is_openai: bool = True):
    """
    Generates a chat response using the smol_dev library by prompting it to create
    the content for a conceptual file "answer.txt".
    This function handles setting up the environment for OpenAI or (experimentally) Ollama,
    constructs a specialized prompt, invokes smol_dev's planning and generation steps,
    and attempts to extract a clean textual answer.

    Args:
        user_query (str): The user's input question.
        rag_context (str, optional): Context retrieved from documents for RAG.
        selected_model_name (str, optional): The name of the LLM to be used (OpenAI or Ollama).
        is_openai (bool): True if an OpenAI model is selected, False for Ollama.

    Returns:
        str: The generated textual response or an error message if issues occur.
    """
    if not SMOL_DEV_AVAILABLE:
        # This message is crucial if smol_dev is not installed.
        # The dummy functions also raise NotImplementedError, which should be caught here if called directly,
        # but this check prevents deeper calls if the import itself failed.
        return "Error: smol_dev library is not available. Please ensure it's installed (e.g., `pip install smol_dev`)."

    # 1. Configure LLM environment for smol_dev
    # This step is critical for directing smol_dev to the correct LLM.
    if is_openai:
        if not ORIGINAL_OPENAI_API_KEY: # Check if original key was available at app start
            return "Error: OPENAI_API_KEY was not found in the environment when the app started. Cannot use OpenAI with smol_dev."
        set_smol_dev_env_for_openai(ORIGINAL_OPENAI_API_KEY, selected_model_name)
    else: # Ollama (Highly Experimental Path)
        # Warn user about the experimental nature of using Ollama with smol_dev.
        st.sidebar.warning(
            f"Attempting to use smol_dev with Ollama model '{selected_model_name}'. "
            "This is highly experimental and relies on Ollama's OpenAI API compatibility."
        )
        set_smol_dev_env_for_ollama(selected_model_name) # Default base: http://localhost:11434/v1

    response_content = ""
    try:
        # 2. Construct the Master Prompt for smol_dev
        # The prompt is engineered to make smol_dev act like a text generator for a single file, "answer.txt".
        # It explicitly tells smol_dev to focus on the answer and avoid code-generation boilerplate.
        master_prompt = f"""You are an AI assistant. Your primary goal is to provide a direct, helpful, and informative textual answer to the user's query.
Imagine you are creating the content for a single file named 'answer.txt'.
The entire output you generate should be the content of this 'answer.txt' file.
Do NOT generate any other file names, code structures, project plans, or explanations about your file generation process.
Focus SOLELY on composing the textual answer to the user's query.
The answer should be well-formatted plain text, suitable for display in a chat interface. Avoid markdown if not essential for the answer.

User Query:
{user_query}
"""
        if rag_context:
            master_prompt += f"""

Available Context (use this to inform your answer if relevant; if not relevant, please ignore it):
---
{rag_context}
---
"""
        master_prompt += "\nYour final output is the content for 'answer.txt'. It should be JUST the answer text, without any additional explanations or conversational fluff."

        # 3. Call smol_dev.plan()
        # This step generates a plan based on the master prompt. For our use case,
        # the "plan" might be simple, but it's part of the smol_dev workflow.
        # Uncomment st.info lines for detailed debugging of smol_dev's internal state.
        # st.info(f"smol_dev: Using prompt for plan (first 300 chars): {master_prompt[:300]}...")
        shared_deps = plan(master_prompt)
        # st.info(f"smol_dev: Plan received (first 300 chars): {shared_deps[:300]}...")

        # 4. Call smol_dev.specify_file_paths()
        # We expect "answer.txt" due to our specific prompting.
        # However, smol_dev might suggest other paths or none.
        file_paths = specify_file_paths(master_prompt, shared_deps)
        # st.info(f"smol_dev: File paths specified by smol_dev: {file_paths}")

        # Determine the target file path for generation.
        # Our prompt strongly suggests 'answer.txt', so we use it as a conceptual target.
        conceptual_file_path = "answer.txt"
        if not file_paths:
            st.warning("smol_dev did not specify any output file paths. Will attempt to generate content for the conceptual 'answer.txt'.")
        elif conceptual_file_path not in file_paths:
            # If smol_dev suggests paths but not "answer.txt", we log this and pick the first one it suggests,
            # hoping our prompt was strong enough to make it write the answer there.
            st.warning(
                f"smol_dev did not explicitly specify '{conceptual_file_path}'. "
                f"Using the first path from smol_dev: '{file_paths[0]}' if available, otherwise falling back to '{conceptual_file_path}'."
            )
            conceptual_file_path = file_paths[0] if file_paths else conceptual_file_path


        # 5. Call smol_dev.generate_code_sync()
        # This is where smol_dev calls the LLM to generate the content for the specified file path.
        # We expect this content to be our textual answer.
        # st.info(f"smol_dev: Requesting generation for target file path: {conceptual_file_path}...")
        generated_output = generate_code_sync(master_prompt, shared_deps, conceptual_file_path)

        if generated_output is None:
             response_content = "Error: smol_dev returned an empty response (None). This might indicate an issue with the LLM configuration, the model itself, or restrictive content filters."
        else:
            response_content = generated_output.strip()
            # Post-processing: smol_dev might still wrap the output in boilerplate (e.g., "Writing to file...").
            # The following lines attempt to clean common boilerplate patterns. More sophisticated cleaning might be needed
            # depending on observed `smol_dev` behavior with different models.
            lines = response_content.splitlines()
            if lines and "writing to" in lines[0].lower() and (conceptual_file_path in lines[0].lower() or ".txt" in lines[0].lower()):
                # If the first line looks like "Writing to answer.txt...", remove it.
                response_content = "\n".join(lines[1:]).strip()

            # Remove potential markdown code blocks if smol_dev wraps simple text in them.
            if response_content.startswith("```text\n") and response_content.endswith("\n```"):
                 response_content = response_content.removeprefix("```text\n").removesuffix("\n```").strip()
            elif response_content.startswith("```\n") and response_content.endswith("\n```"): # Generic code block
                 response_content = response_content.removeprefix("```\n").removesuffix("\n```").strip()


            if not response_content: # If stripping boilerplate results in an empty string
                response_content = ("smol_dev generated a response, but it appears to be empty after attempting to clean "
                                    "common boilerplate. The raw output might have been just boilerplate or unintended formatting.")

    except NotImplementedError:
        # This error is specifically raised by our dummy smol_dev functions if the library wasn't imported.
        # The st.error message would have already been shown by the dummy function.
        response_content = "Error: smol_dev functions are not available. Please ensure the 'smol_dev' library is correctly installed."
    except Exception as e:
        st.error(f"An unexpected error occurred during smol_dev processing: {e}")
        # Provide a more generic error to the user but log/print details for debugging.
        # print(f"Full smol_dev error: {type(e).__name__} - {e}", file=sys.stderr) # Consider logging properly
        response_content = f"Sorry, an error occurred while trying to generate the response using smol_dev. Details: {str(e)}"
    finally:
        # 6. CRITICAL: Restore original environment variables
        # This prevents interference with subsequent operations or other parts of the application.
        restore_original_env()
        # st.info("smol_dev: Original environment variables restored.") # For debugging

    return response_content

# --- RAG Pipeline Functions ---
def load_document(uploaded_file):
    """
    Loads an uploaded file (PDF, TXT, MD) and returns its text content.
    Uses a temporary file to handle the uploaded file object for loaders.
    """
    if not uploaded_file:
        return None

    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    text_content = ""
    tmp_file_path = "" # Initialize to ensure it's available for finally block

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        if file_ext == ".pdf":
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            text_content = "\n".join([doc.page_content for doc in documents])
        elif file_ext in [".txt", ".md"]:
            loader = TextLoader(tmp_file_path, encoding='utf-8')
            documents = loader.load()
            if documents:
                 text_content = documents[0].page_content
        else:
            st.error(f"Unsupported file type: {file_ext}")
            return None

    except Exception as e:
        st.error(f"Error loading document '{uploaded_file.name}': {e}")
        return None
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

    return text_content

def get_text_chunks(text: str):
    """Splits text into manageable chunks."""
    if not text:
        return []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks: list, embedding_model_name: str, is_openai_embedding: bool):
    """
    Creates a vector store from text chunks using appropriate embeddings.
    Args:
        text_chunks: List of text chunks.
        embedding_model_name: Name of the embedding model to use (e.g., "gpt-3.5-turbo" for OpenAI default, or an Ollama model name).
        is_openai_embedding: True if using OpenAI embeddings, False for Ollama.
    Returns:
        A Chroma retriever instance or None if an error occurs.
    """
    if not text_chunks:
        st.warning("No text chunks to process for vector store.")
        return None

    embeddings = None
    try:
        if is_openai_embedding:
            if not os.getenv("OPENAI_API_KEY"):
                st.error("OpenAI API key is not set. Cannot create embeddings for OpenAI models.")
                return None
            # For OpenAI, embedding_model_name can be specified if needed, e.g., "text-embedding-ada-002"
            # Using default OpenAIEmbeddings behavior if a generic chat model name is passed.
            embeddings = OpenAIEmbeddings()
        else: # Ollama embeddings
            # Here, embedding_model_name should be a specific Ollama model that can provide embeddings.
            # e.g., "nomic-embed-text", "mxbai-embed-large", or even a general model like "llama3" if it's capable.
            # The user must ensure the specified Ollama model in the UI is suitable for embeddings.
            import ollama # Import here to avoid error if ollama is not configured and this path isn't taken.
            try:
                # Check if the ollama server is running and the model is available.
                # This doesn't guarantee it's a good *embedding* model, but it's a basic check.
                ollama.list() # Basic connection check
                embeddings = OllamaEmbeddings(model=embedding_model_name)
            except Exception as e:
                st.error(f"Failed to initialize or connect to Ollama for embeddings with model '{embedding_model_name}': {e}")
                st.info("Ensure Ollama is running and the selected model supports embeddings or choose a dedicated embedding model.")
                return None

        if embeddings is None:
             st.error("Failed to initialize any embedding model.")
             return None

        vector_store = Chroma.from_texts(texts=text_chunks, embedding=embeddings)
        return vector_store.as_retriever()

    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

# --- Ollama Integration (for model listing) ---
def get_ollama_models_list(): # Renamed to avoid conflict if an old get_ollama_models existed
    """
    Connects to the Ollama API and fetches the list of locally available models.
    Returns a list of model names. Handles connection errors gracefully.
    """
    try:
        client = ollama.Client()
        models_info = client.list()
        model_names = [model['name'] for model in models_info['models']]
        return model_names
    except Exception: # Catch broad exception if Ollama is not running or accessible
        return []


# --- Main Streamlit App ---
def main():
    st.title("Chatbot with smol_dev and RAG")

    # --- Sidebar ---
    with st.sidebar:
        st.header("Configuration")

        # Model Selection
        openai_models = ["gpt-4o", "gpt-3.5-turbo"] # Add more as needed

        local_ollama_models = get_ollama_models_list()
        if not local_ollama_models:
            ollama_display_option = ["Ollama N/A (No models or connection error)"]
            available_models = openai_models + ollama_display_option
        else:
            available_models = openai_models + local_ollama_models

        selected_model_name = st.selectbox("Choose a Model", available_models)

        is_openai_selected = selected_model_name in openai_models

        # For RAG embeddings, we'll use the type of the selected chat model.
        # If an Ollama model is chosen for chat, we'll attempt to use it (or a related one) for embeddings.
        # This simplifies the UI but assumes the chosen Ollama model can also serve as an embedding model
        # or that OllamaEmbeddings can handle it. This is often not ideal for Ollama.
        # A dedicated embedding model selector for Ollama would be more robust.
        embedding_model_for_rag = selected_model_name
        is_openai_embedding_model = is_openai_selected

        st.info(f"""**Note on smol_dev & Models:**
        - OpenAI models are expected to work with `smol_dev`.
        - Ollama model usage with `smol_dev` is **experimental** and relies on Ollama mimicking the OpenAI API and `smol_dev` respecting environment variables for API base. This may not work as `smol_dev` is not designed for it.
        - For RAG with Ollama, the selected Ollama chat model ('{selected_model_name}') will also be attempted for embeddings. This might not be optimal; dedicated embedding models are usually better.
        """)

        # File Uploader for RAG
        uploaded_file = st.file_uploader("Upload a document for RAG", type=['txt', 'md', 'pdf'], key="file_uploader")

        if "retriever" not in st.session_state:
            st.session_state.retriever = None
        if "processed_file_name" not in st.session_state:
            st.session_state.processed_file_name = None

        if uploaded_file:
            if st.session_state.processed_file_name != uploaded_file.name:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    raw_text = load_document(uploaded_file)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        if text_chunks:
                            st.session_state.retriever = get_vector_store(
                                text_chunks,
                                embedding_model_name=embedding_model_for_rag,
                                is_openai_embedding=is_openai_embedding_model
                            )
                            if st.session_state.retriever:
                                st.session_state.processed_file_name = uploaded_file.name
                                st.success(f"Document '{uploaded_file.name}' processed for RAG.")
                            else:
                                st.error(f"Could not create RAG retriever for '{uploaded_file.name}'. See logs if any.")
                                st.session_state.processed_file_name = None # Reset on failure
                        else:
                            st.warning(f"No text chunks extracted from '{uploaded_file.name}'. Cannot set up RAG.")
                            st.session_state.processed_file_name = None
                    else:
                        st.error(f"Failed to load document: '{uploaded_file.name}'.")
                        st.session_state.processed_file_name = None
            elif st.session_state.retriever:
                st.success(f"Document '{uploaded_file.name}' is already processed and ready for RAG.")

        if st.session_state.retriever and st.button("Clear Loaded Document"):
            st.session_state.retriever = None
            st.session_state.processed_file_name = None
            st.rerun() # Rerun to update UI after clearing


    # --- Main Chat Interface ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is your question?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = "Thinking..." # Initial placeholder message
            response_placeholder.markdown(full_response + "▌")

            rag_context_str = None
            if st.session_state.retriever:
                try:
                    # Retrieve context based on the user's prompt
                    retrieved_docs = st.session_state.retriever.invoke(prompt)
                    rag_context_str = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
                    if rag_context_str:
                        # Display a snippet of the RAG context in the sidebar for transparency/debugging
                        with st.sidebar.expander("Retrieved RAG Context Snippet"):
                            st.text(rag_context_str[:500] + "..." if len(rag_context_str) > 500 else rag_context_str)
                    else:
                        st.sidebar.info("RAG: No specific context found for this query.")
                except Exception as e:
                    st.sidebar.warning(f"RAG retrieval failed: {e}")

            # Check for unavailable models or smol_dev library before calling response generation
            if selected_model_name == "Ollama N/A (No models or connection error)":
                full_response = "Cannot process request: Ollama is not available or no models were found. Please check your Ollama setup."
            elif not SMOL_DEV_AVAILABLE:
                 full_response = "Critical Error: smol_dev library is not installed or failed to import. This app requires smol_dev to function."
            else:
                # Generate response using smol_dev, potentially with RAG context
                full_response = get_smol_dev_response(
                    user_query=prompt,
                    rag_context=rag_context_str,
                    selected_model_name=selected_model_name,
                    is_openai=is_openai_selected
                )

            response_placeholder.markdown(full_response) # Display final response

        # Store the assistant's response in session state
        st.session_state.messages.append({"role": "assistant", "content": full_response if full_response else "No response generated or an error occurred."})

if __name__ == "__main__":
    main()
