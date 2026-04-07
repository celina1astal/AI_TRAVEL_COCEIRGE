import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage, AIMessage

# --- 1. PAGE CONFIG & SECRETS ---
st.set_page_config(page_title="AI Travel Agent", page_icon="✈️")
st.title("✈️ AI Travel Agent")

# Get Keys from Streamlit Secrets
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY")

# Set Tavily Key in Environment for the tool to find it
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY or ""

# --- 2. PDF DATA LOADING (CACHED) ---
@st.cache_resource
def load_data():
    if not os.path.exists("travel_sample.pdf"):
        st.error("Missing 'travel_sample.pdf'!")
        return None
    
    loader = PyPDFLoader("travel_sample.pdf")
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = splitter.split_documents(pages)
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=GEMINI_API_KEY
    )
    return FAISS.from_documents(docs, embeddings)

vector_db = load_data()

# --- 3. TOOL DEFINITIONS ---

@tool
def search_travel_pdf(query: str):
    """Searches the internal travel manual for specific flights or plans."""
    docs = vector_db.similarity_search(query, k=3)
    return "\n".join([d.page_content for d in docs])

# New Tools
web_search = TavilySearchResults(max_results=2)
wiki_api = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=800)
wiki_search = WikipediaQueryRun(api_wrapper=wiki_api)

# Tool Registry
tools = [search_travel_pdf, web_search, wiki_search]
tool_map = {
    "search_travel_pdf": search_travel_pdf,
    "tavily_search_results_json": web_search,
    "wikipedia": wiki_search
}

# --- 4. LLM SETUP ---
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY).bind_tools(tools)

# --- 5. SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a professional travel agent. Use tools for specific info.")
    ]

# Display History
for m in st.session_state.messages:
    if isinstance(m, HumanMessage):
        st.chat_message("user").write(m.content)
    elif isinstance(m, AIMessage) and m.content:
        st.chat_message("assistant").write(m.content)

# --- 6. AGENT LOOP WITH ERROR HANDLING ---
if user_query := st.chat_input("Ask me anything..."):
    st.chat_message("user").write(user_query)
    st.session_state.messages.append(HumanMessage(content=user_query))

    with st.chat_message("assistant"):
        response = llm.invoke(st.session_state.messages)
        
        if hasattr(response, 'tool_calls') and response.tool_calls:
            st.session_state.messages.append(response)
            
            for tool_call in response.tool_calls:
                t_name = tool_call["name"]
                t_args = tool_call["args"]
                
                with st.status(f"Executing: {t_name}...") as status:
                    try:
                        if t_name in tool_map:
                            result = tool_map[t_name].invoke(t_args)
                            status.update(label=f"{t_name} Success!", state="complete")
                        else:
                            result = f"Error: Tool '{t_name}' not recognized."
                            status.update(label="Tool Not Found", state="error")
                    except Exception as e:
                        result = f"API Error: {str(e)}"
                        status.update(label="Execution Failed", state="error")
                
                st.session_state.messages.append(
                    ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                )
            
            # Final Summary Call
            response = llm.invoke(st.session_state.messages)

        if response.content:
            st.markdown(response.content)
            st.session_state.messages.append(response)
