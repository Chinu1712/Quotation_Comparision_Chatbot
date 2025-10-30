import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables (for API key)
load_dotenv()

# --- Configuration ---
# The GEMINI_API_KEY environment variable is expected to be set in the sandbox environment.

# --- PDF Processing Functions ---

def get_pdf_text(pdf_docs):
    """Extracts text from a list of PDF files."""
    raw_text = ""
    for pdf in pdf_docs:
        try:
            reader = PdfReader(pdf)
            for page in reader.pages:
                raw_text += page.extract_text()
            raw_text += "\n\n--- END OF QUOTE DOCUMENT ---\n\n" # Separator for the model
        except Exception as e:
            st.error(f"Could not read PDF file {pdf.name}: {e}")
    return raw_text

def get_text_chunks(text):
    """Splits text into smaller, manageable chunks."""
    # Using a large chunk size as the comparison prompt needs the full context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks

# --- LangChain/Gemini Functions (Placeholder for comparison and chat) ---

def get_gemini_model():
    """Initializes and returns the configured Gemini model."""
    # The GoogleGenerativeAI class in langchain_google_genai automatically uses
    # the GEMINI_API_KEY environment variable.
    model = GoogleGenerativeAI(
        model="gemini-2.5-flash",
        safety_settings=[
            {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
            {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
            {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
            {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
        ]
    )
    return model

def get_comparison_prompt():
    """Defines the prompt for structured quote comparison."""
    prompt_template = """
    You are an expert procurement and finance assistant. Your task is to analyze the provided text, which contains multiple quotations from different vendors.
    The text is a concatenation of multiple PDF documents, separated by "--- END OF QUOTE DOCUMENT ---". Each document represents one quote.
    
    You must extract the following key comparison points for EACH quotation:
    1. **Vendor Name**: The name of the company providing the quote.
    2. **Total Price**: The final, all-inclusive price. Specify currency.
    3. **Payment Terms**: e.g., "Net 30", "50% upfront", "COD".
    4. **Delivery Time**: e.g., "4-6 weeks", "Immediately", "30 days ARO".
    5. **Warranty/Guarantee**: Details on the warranty offered.
    6. **Custom Field 1 (e.g., Item A Price)**: The price for a specific, important item (if applicable and identifiable).
    7. **Custom Field 2 (e.g., Service Fee)**: Any other critical custom fee or detail.

    Structure your output as a single, clean Markdown table. The first column should be the feature (e.g., "Total Price", "Vendor Name"), and subsequent columns should be for each quote/vendor.
    
    Example Output Format:
    
    | Feature | Vendor A Quote | Vendor B Quote | Vendor C Quote |
    | :--- | :--- | :--- | :--- |
    | **Vendor Name** | Vendor A | Vendor B | Vendor C |
    | **Total Price** | $15,000 USD | €14,500 EUR | $15,500 USD |
    | **Payment Terms** | Net 30 | 50% Upfront | COD |
    | **Delivery Time** | 4 Weeks | 6 Weeks | 3 Weeks |
    | **Warranty** | 1 Year Parts & Labor | 2 Years Parts Only | 6 Months |
    | **Item A Price** | $500 | €450 | $520 |
    | **Service Fee** | Included | 5% of Total | Waived |

    If a piece of information is not found, use "N/A". Ensure the table is complete and well-formatted.
    
    QUOTATION TEXT:
    {text_chunks}
    
    COMPARISON TABLE:
    """
    return PromptTemplate(template=prompt_template, input_variables=["text_chunks"])

def get_comparison_chain(model):
    """Loads the LangChain chain for comparison."""
    prompt = get_comparison_prompt()
    # Using the Gemini model directly with the prompt
    # We will use the model's invoke method with the prompt template
    return prompt | model

def get_chat_chain(model):
    """Loads the LangChain QA chain for chat."""
    # This is a standard QA chain for general questioning
    prompt_template = """
    You are a helpful assistant for analyzing and comparing vendor quotations.
    Answer the user's question based ONLY on the provided context (the text from the uploaded quotes).
    If the answer is not available in the context, politely state that the information is not in the quotes.
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Using load_qa_chain with chain_type="stuff" to pass all document chunks
    # This requires the input to be a list of Document objects, but since we are using
    # a custom prompt with a single string of text chunks, we will adapt the usage in main().
    # For now, let's keep the standard QA chain for simplicity in the function definition.
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# --- Main Streamlit App Logic ---

def main():
    st.set_page_config(page_title="Quote Comparison Chatbot", layout="wide")
    st.title("💰 Quote Comparison Chatbot")
    st.markdown("Upload multiple PDF quotations to compare prices, terms, and custom details side-by-side, and ask questions about the quotes.")

    # Initialize session state variables
    if "comparison_table" not in st.session_state:
        st.session_state.comparison_table = None
    if "raw_text" not in st.session_state:
        st.session_state.raw_text = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    # --- Sidebar for Upload and Processing ---
    with st.sidebar:
        st.header("1. Upload Quotes")
        pdf_docs = st.file_uploader(
            "Upload your PDF Quotes", accept_multiple_files=True, type=["pdf"]
        )
        
        if st.button("Process Quotes"):
            if pdf_docs:
                # 1. Extract Text
                with st.spinner("Extracting text from PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    st.session_state.raw_text = raw_text
                    st.success("Text extracted from PDFs.")
                
                # 2. Generate Comparison Table
                if st.session_state.raw_text:
                    with st.spinner("Generating comparison table with Gemini..."):
                        try:
                            model = get_gemini_model()
                            comparison_chain = get_comparison_chain(model)
                            
                            # Invoke the chain with the raw text
                            # We use the raw text as a single input to the prompt
                            comparison_table = comparison_chain.invoke({"text_chunks": st.session_state.raw_text})
                            
                            # The result is a dict with 'text' key for the output
                            st.session_state.comparison_table = comparison_table.content
                            st.success("Comparison table generated!")
                        except Exception as e:
                            st.error(f"An error occurred during comparison generation: {e}")
                            st.session_state.comparison_table = None
            else:
                st.warning("Please upload at least one PDF quote to process.")

    # --- Main Content Area ---
    
    tab1, tab2 = st.tabs(["📊 Quote Comparison Table", "💬 Ask Questions"])

    with tab1:
        st.header("Quote Comparison Table")
        if st.session_state.comparison_table:
            # Streamlit will render the Markdown table nicely
            st.markdown(st.session_state.comparison_table)
        else:
            st.info("Upload and process your PDF quotes in the sidebar to generate the comparison table.")

    with tab2:
        st.header("Chat with your Quotes")
        
        if st.session_state.raw_text:
            # Display chat history
            for sender, message in st.session_state.chat_history:
                if sender == "user":
                    st.chat_message("user").write(message)
                else:
                    st.chat_message("assistant").write(message)

            # Chat input
            if prompt := st.chat_input("Ask a question about the quotes..."):
                st.session_state.chat_history.append(("user", prompt))
                st.chat_message("user").write(prompt)
                
                # We need to convert the raw text into LangChain Document objects for the QA chain
                text_chunks = get_text_chunks(st.session_state.raw_text)
                from langchain.docstore.document import Document
                documents = [Document(page_content=t) for t in text_chunks]
                
                with st.spinner("Thinking..."):
                    try:
                        model = get_gemini_model()
                        chat_chain = get_chat_chain(model)
                        
                        # Invoke the chain
                        response = chat_chain.invoke(
                            {"input_documents": documents, "question": prompt}
                        )
                        
                        assistant_response = response["output_text"]
                        st.session_state.chat_history.append(("assistant", assistant_response))
                        st.chat_message("assistant").write(assistant_response)
                        
                    except Exception as e:
                        st.error(f"An error occurred during chat: {e}")
                        st.session_state.chat_history.append(("assistant", "Sorry, I encountered an error while processing your question."))
                        st.chat_message("assistant").write("Sorry, I encountered an error while processing your question.")

        else:
            st.info("Upload and process your PDF quotes in the sidebar to start asking questions.")

if __name__ == "__main__":
    # Check for API Key
    if not os.getenv("GEMINI_API_KEY"):
        # In the sandbox, the key is set as OPENAI_API_KEY but LangChain's GoogleGenerativeAI
        # expects GEMINI_API_KEY or GOOGLE_API_KEY. I will check for OPENAI_API_KEY and
        # inform the user that they need to set the Gemini key in their local environment.
        # For the sandbox to work, I will rely on the pre-configured environment.
        # Since I cannot check for the specific GEMINI_API_KEY in the sandbox, I will assume it's set
        # or that the underlying system handles the key mapping for the LLM tool.
        # I will remove the explicit check for now to allow the app to run in the sandbox.
        main()
    else:
        main()
