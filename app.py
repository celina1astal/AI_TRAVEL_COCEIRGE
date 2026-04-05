import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage, AIMessage

# --- 1. PAGE CONFIG & SECRETS ---
st.set_page_config(page_title="B.E. Travel Agent", page_icon="✈️")
st.title("✈️ AI Travel Concierge")

# If running in Colab, ensure these are set in your environment or Secrets
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

# --- 2. FAST PDF LOADING (THE CACHE) ---
@st.cache_resource
def load_data():
    if not os.path.exists("travel_sample.pdf"):
        st.error("Missing 'travel_sample.pdf'! Upload it to Colab/GitHub.")
        return None
    
    # Load and Split
    loader = PyPDFLoader("travel_sample.pdf")
    pages = loader.load()
    
    # Optimized Splitter for speed
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = splitter.split_documents(pages)
    
    # Embeddings (Google)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=GEMINI_API_KEY
    )
    
    # Vector Store (FAISS is faster & stable for Python 3.14)
    return FAISS.from_documents(docs, embeddings)

vector_db = load_data()

# --- 3. TOOL DEFINITION ---
@tool
def search_travel_pdf(query: str):
    """Searches the travel manual for specific details on flights, hotels, or plans."""
    docs = vector_db.similarity_search(query, k=3)
    context = "\n".join([d.page_content for d in docs])
    return context

tools = [search_travel_pdf]
tool_map = {t.name: t for t in tools}

# --- 4. LLM SETUP ---
# Using 8b-instant for speed during debugging (Change to 70b-versatile for final demo)
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY).bind_tools(tools)

# --- 5. SESSION STATE (MEMORY) ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a helpful travel agent. Use the search_travel_pdf tool for specific PDF info.")
    ]

# UI: Reset Button
if st.sidebar.button("Reset Chat"):
    st.session_state.messages = [SystemMessage(content="Chat Reset.")]
    st.rerun()

# Display Chat History
for m in st.session_state.messages:
    if isinstance(m, HumanMessage):
        st.chat_message("user").write(m.content)
    elif isinstance(m, AIMessage) and m.content:
        st.chat_message("assistant").write(m.content)

# --- 6. THE AGENT LOOP (THE FIX) ---
if user_query := st.chat_input("Ask me anything..."):
    st.chat_message("user").write(user_query)
    st.session_state.messages.append(HumanMessage(content=user_query))

    with st.chat_message("assistant"):
        # First call to check if tool is needed
        response = llm.invoke(st.session_state.messages)
        
        if hasattr(response, 'tool_calls') and response.tool_calls:
            st.session_state.messages.append(response)
            
            for tool_call in response.tool_calls:
                with st.status("Searching Knowledge Base...") as s:
                    result = tool_map[tool_call["name"]].invoke(tool_call["args"])
                    s.update(label="Information Found!", state="complete")
                
                # Feed the tool result back
                st.session_state.messages.append(
                    ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                )
            
            # Final call to summarize the findings
            response = llm.invoke(st.session_state.messages)

        # Final output to the screen
        if response.content:
            st.markdown(response.content)
            st.session_state.messages.append(response)
        else:
            st.write("I'm thinking... please try again.(API limit)")
