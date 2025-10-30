import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from dotenv import load_dotenv
import os

# Load environment variables (for API key)
load_dotenv()

# --- PDF Processing Functions ---

def get_pdf_text(pdf_docs):
    """Extracts text from a list of PDF files."""
    raw_text = ""
    for pdf in pdf_docs:
        try:
            reader = PdfReader(pdf)
            for page in reader.pages:
                if page.extract_text():
                    raw_text += page.extract_text()
            raw_text += "\n\n--- END OF QUOTE DOCUMENT ---\n\n"
        except Exception as e:
            st.error(f"Could not read PDF file {pdf.name}: {e}")
    return raw_text


def get_text_chunks(text):
    """Splits long text into smaller, manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    return text_splitter.split_text(text)

# --- Gemini Model Setup ---

def get_gemini_model():
    """Initializes and returns a configured Gemini model."""
    return GoogleGenerativeAI(
        model="gemini-2.5-flash",
        safety_settings=[
            {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
            {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
            {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
            {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
        ]
    )

# --- Prompt Definitions ---

def get_comparison_prompt():
    """Prompt for structured vendor quote comparison."""
    prompt_template = """
    You are an expert procurement and finance assistant. Your task is to analyze the provided text, 
    which contains multiple quotations from different vendors. The text is a concatenation of multiple 
    PDF documents, separated by "--- END OF QUOTE DOCUMENT ---". Each document represents one quote.

    Extract the following key comparison points for EACH quotation:
    1. **Vendor Name**
    2. **Total Price** (include currency)
    3. **Payment Terms**
    4. **Delivery Time**
    5. **Warranty/Guarantee**
    6. **Custom Field 1 (e.g., Item A Price)**
    7. **Custom Field 2 (e.g., Service Fee)**

    Present your output as a Markdown table with features in rows and vendors in columns.

    If any information is missing, write "N/A".

    QUOTATION TEXT:
    {text_chunks}

    COMPARISON TABLE:
    """
    return PromptTemplate(template=prompt_template, input_variables=["text_chunks"])


def get_comparison_chain(model):
    """Creates a chain for structured comparison."""
    prompt = get_comparison_prompt()
    return prompt | model


def get_chat_chain(model):
    """Creates a chain for follow-up Q&A."""
    prompt_template = """
    You are a helpful assistant for analyzing and comparing vendor quotations.
    Answer the user's question based ONLY on the provided context (the uploaded quote text).
    If the answer is not available, say "The information is not available in the quotes."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# --- Streamlit App ---

def main():
    st.set_page_config(page_title="💰 Quote Comparison Chatbot", layout="wide")
    st.title("💰 Quote Comparison Chatbot")
    st.markdown("Upload multiple vendor quotations (PDFs) to compare prices, terms, and details — then ask questions about them!")

    # --- Initialize Session State ---
    if "comparison_table" not in st.session_state:
        st.session_state.comparison_table = None
    if "raw_text" not in st.session_state:
        st.session_state.raw_text = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # --- Sidebar ---
    with st.sidebar:
        st.header("📂 Upload and Process Quotes")
        pdf_docs = st.file_uploader("Upload your PDF quotes", accept_multiple_files=True, type=["pdf"])

        if st.button("⚙️ Process Quotes"):
            if pdf_docs:
                with st.spinner("🔍 Extracting text from PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    st.session_state.raw_text = raw_text
                    st.success("✅ Text extracted successfully!")

                if st.session_state.raw_text:
                    with st.spinner("🤖 Generating comparison table with Gemini..."):
                        try:
                            model = get_gemini_model()
                            chain = get_comparison_chain(model)
                            result = chain.invoke({"text_chunks": st.session_state.raw_text})

                            # Handle both string and object outputs safely
                            st.session_state.comparison_table = getattr(result, "content", str(result))
                            st.success("✅ Comparison table generated!")
                        except Exception as e:
                            st.error(f"⚠️ Error during comparison generation: {e}")
                            st.session_state.comparison_table = None
            else:
                st.warning("⚠️ Please upload at least one PDF file first.")

        if st.button("🔄 Reset App"):
            for key in ["comparison_table", "raw_text", "chat_history"]:
                st.session_state[key] = None if key != "chat_history" else []
            st.rerun()

    # --- Tabs ---
    tab1, tab2 = st.tabs(["📊 Quote Comparison", "💬 Chat with Quotes"])

    # --- Tab 1: Comparison ---
    with tab1:
        st.header("📊 Quote Comparison Table")
        if st.session_state.comparison_table:
            st.markdown(st.session_state.comparison_table)
        else:
            st.info("Upload and process your PDF quotes to generate a comparison table.")

    # --- Tab 2: Chat ---
    with tab2:
        st.header("💬 Ask Questions About the Quotes")

        if st.session_state.raw_text:
            for sender, msg in st.session_state.chat_history:
                st.chat_message(sender).write(msg)

            if user_input := st.chat_input("Ask a question about the quotes..."):
                st.session_state.chat_history.append(("user", user_input))
                st.chat_message("user").write(user_input)

                with st.spinner("🤔 Thinking..."):
                    try:
                        text_chunks = get_text_chunks(st.session_state.raw_text)
                        documents = [Document(page_content=t) for t in text_chunks]
                        model = get_gemini_model()
                        chat_chain = get_chat_chain(model)
                        response = chat_chain.invoke({"input_documents": documents, "question": user_input})

                        assistant_reply = response.get("output_text", str(response))
                        st.session_state.chat_history.append(("assistant", assistant_reply))
                        st.chat_message("assistant").write(assistant_reply)
                    except Exception as e:
                        st.error(f"⚠️ Chat error: {e}")
                        st.session_state.chat_history.append(("assistant", "Sorry, I encountered an error."))
        else:
            st.info("Please upload and process PDF quotes first to start chatting.")

if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        st.warning("⚠️ GEMINI_API_KEY not found in your environment. Please check your .env file.")
    main()
