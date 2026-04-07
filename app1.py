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

# --- 1. CONFIG & SECRETS ---
st.set_page_config(page_title="AI Travel Agent", page_icon="✈️", layout="wide")

# Fetch Keys from Secrets
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY")
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY or ""

# --- 2. KNOWLEDGE BASE INITIALIZATION ---
@st.cache_resource
def load_vector_db():
    if not os.path.exists("travel_sample.pdf"):
        st.error("File 'travel_sample.pdf' not found!")
        return None
    
    try:
        loader = PyPDFLoader("travel_sample.pdf")
        pages = loader.load()
        # Increase chunk size to 3000 to reduce the number of API calls (avoids rate limits)
        splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
        docs = splitter.split_documents(pages)
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=GEMINI_API_KEY # Uses the key fetched at the top
        )
        
        # This is where the error happens - we wrap it in a try/except
        return FAISS.from_documents(docs, embeddings)
    
    except Exception as e:
        st.error(f"❌ Google AI Error: {str(e)}")
        # This allows the app to keep running even if the PDF part fails
        return None

# --- 3. TOOL DEFINITIONS ---

@tool
def search_travel_pdf(query: str):
    """Searches the internal travel manual for specific flight codes or plans."""
    if vector_db is not None:
        docs = vector_db.similarity_search(query, k=3)
        return "\n".join([d.page_content for d in docs])
    return "Local knowledge base is currently unavailable."

# Modern Tavily & Wikipedia tools
web_search = TavilySearch(max_results=2)
wiki_api = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=800)
wiki_search = WikipediaQueryRun(api_wrapper=wiki_api)

# --- 4. THE TOOL REGISTRY (Fixed NameError Sequence) ---
# All tools are now defined above, so this map will NOT throw a NameError
tools = [search_travel_pdf, web_search, wiki_search]

tool_map = {
    "search_travel_pdf": search_travel_pdf,
    "tavily_search_results_json": web_search, # Name LLM uses for Tavily
    "wikipedia": wiki_search
}

# --- 5. LLM SETUP ---
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY).bind_tools(tools)

# --- 6. UI & CHAT LOGIC ---
st.title("✈️ AI Travel Agent")

if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content="You are a professional travel agent.")]

# Display Chat History
for m in st.session_state.messages:
    if isinstance(m, HumanMessage):
        st.chat_message("user").write(m.content)
    elif isinstance(m, AIMessage) and m.content:
        st.chat_message("assistant").write(m.content)

# Chat Input Loop
if user_query := st.chat_input("Ask about your trip..."):
    st.chat_message("user").write(user_query)
    st.session_state.messages.append(HumanMessage(content=user_query))

    with st.chat_message("assistant"):
        response = llm.invoke(st.session_state.messages)
        
        # Check if AI needs to use tools
        if hasattr(response, 'tool_calls') and response.tool_calls:
            st.session_state.messages.append(response)
            
            for tool_call in response.tool_calls:
                t_name = tool_call["name"]
                t_args = tool_call["args"]
                
                with st.status(f"Running {t_name}...") as status:
                    if t_name in tool_map:
                        result = tool_map[t_name].invoke(t_args)
                        status.update(label=f"{t_name} Success!", state="complete")
                    else:
                        result = f"Error: Tool {t_name} not recognized."
                        status.update(label="Tool Not Found", state="error")
                
                st.session_state.messages.append(
                    ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                )
            
            # Final summary after tools
            response = llm.invoke(st.session_state.messages)

        if response.content:
            st.markdown(response.content)
            st.session_state.messages.append(response)
