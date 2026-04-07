import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage, AIMessage

# --- 1. PAGE CONFIG & SECRETS ---
st.set_page_config(page_title="AI Travel Agent", page_icon="✈️", layout="wide")

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY")
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY or ""

# --- 2. RAG ENGINE (CACHED) ---
@st.cache_resource
def load_vector_db():
    if not os.path.exists("travel_sample.pdf"):
        st.error("Please upload 'travel_sample.pdf' to your GitHub repository.")
        return None
    
    try:
        loader = PyPDFLoader("travel_sample.pdf")
        pages = loader.load()
        # Larger chunks to stay within Google's Free Tier rate limits
        splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
        docs = splitter.split_documents(pages)
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=GEMINI_API_KEY
        )
        return FAISS.from_documents(docs, embeddings)
    except Exception as e:
        st.sidebar.error(f"PDF Vector Error: {e}")
        return None

# Global Variable for the Database
vector_db = load_vector_db()

# --- 3. TOOL DEFINITIONS ---

@tool
def search_travel_pdf(query: str):
    """Searches the internal travel manual for specific itineraries or flight codes."""
    if vector_db:
        docs = vector_db.similarity_search(query, k=3)
        return "\n".join([d.page_content for d in docs])
    return "Knowledge base not initialized. Check API keys."

# Modern Tavily with Advanced Search for better weather/live data
web_search = TavilySearch(max_results=3, search_depth="advanced")

# Wikipedia Setup
wiki_api = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
wiki_search = WikipediaQueryRun(api_wrapper=wiki_api)

# --- 4. THE TOOL REGISTRY (Fixed Syntax & Order) ---
tools = [search_travel_pdf, web_search, wiki_search]

# Mapping aliases so the LLM never gets a 'Tool Not Found' error
tool_map = {
    "search_travel_pdf": search_travel_pdf,
    "tavily_search_results_json": web_search,
    "tavily_search": web_search,
    "wikipedia": wiki_search
}

# --- 5. AGENT SETUP ---
llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    api_key=GROQ_API_KEY
).bind_tools(tools)

# --- 6. CHAT INTERFACE ---
st.title("✈️ AI Travel Concierge")

if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="""You are a professional travel agent. 
        - For greetings like 'Hi', 'Hello', or 'How are you', reply politely WITHOUT using any tools.
        - Use 'search_travel_pdf' for internal flight/plan data.
        - Use 'tavily_search_results_json' for live weather or news.
        - Use 'wikipedia' ONLY for historical landmarks or general geography.""")
    ]

# Display history
for m in st.session_state.messages:
    if isinstance(m, HumanMessage):
        st.chat_message("user").write(m.content)
    elif isinstance(m, AIMessage) and m.content:
        st.chat_message("assistant").write(m.content)

# --- 7. AGENT EXECUTION LOOP ---
if user_input := st.chat_input("Ask me about your trip..."):
    st.chat_message("user").write(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))

    with st.chat_message("assistant"):
        response = llm.invoke(st.session_state.messages)
        
        # Handle Tool Calling
        if hasattr(response, 'tool_calls') and response.tool_calls:
            st.session_state.messages.append(response)
            
            for tool_call in response.tool_calls:
                t_name = tool_call["name"]
                t_args = tool_call["args"]
                
                with st.status(f"Consulting {t_name}...") as status:
                    if t_name in tool_map:
                        result = tool_map[t_name].invoke(t_args)
                        status.update(label=f"Done: {t_name}", state="complete")
                    else:
                        result = f"Error: Tool '{t_name}' not found."
                        status.update(label="Error", state="error")
                
                st.session_state.messages.append(
                    ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                )
            
            # Final summary call
            response = llm.invoke(st.session_state.messages)

        if response.content:
            st.markdown(response.content)
            st.session_state.messages.append(response)
