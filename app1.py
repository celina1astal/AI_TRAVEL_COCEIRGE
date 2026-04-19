import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

# --- 1. CONFIG & CSS ---
st.set_page_config(page_title="✈️ AI Travel Concierge", layout="wide")

st.markdown("""
    <style>
    .stDownloadButton > button { background-color: #28a745 !important; color: white !important; width: 100% !important; }
    .stButton > button { width: 100% !important; }
    </style>
""", unsafe_allow_html=True)

# --- 2. SESSION STATE (Initialize First) ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a travel agent. Cite sources like [Source 1] if using the PDF.")
    ]

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Settings")
    model_choice = st.selectbox("🧠 Model", ["gemini-1.5-flash", "gemini-1.5-pro"])
    temp = st.slider("🔥 Creativity", 0.0, 1.0, 0.4)
    
    st.divider()
    
    def export_chat():
        return "\n".join([f"{m.type}: {m.content}" for m in st.session_state.messages if not isinstance(m, SystemMessage)])

    st.download_button("📥 Download Log", data=export_chat(), file_name="chat.txt")
    if st.button("🗑️ Reset"):
        st.session_state.messages = [st.session_state.messages[0]]
        st.rerun()

# --- 4. API & TOOLS ---
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    TAVILY_KEY = st.secrets["TAVILY_API_KEY"]
except:
    st.error("Check Secrets!")
    st.stop()

@tool
def search_travel_pdf(query: str):
    """Searches local PDF travel documents."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = db.similarity_search(query, k=2)
        return "\n".join([f"[Source {i+1}]: {d.page_content}" for i, d in enumerate(docs)])
    except Exception as e:
        return f"PDF Error: {str(e)}"

# Define the Tool List
tools = [search_travel_pdf, TavilySearchResults(tavily_api_key=TAVILY_KEY), WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())]
# Gemini is picky about names. We map them manually.
tool_map = {
    "search_travel_pdf": search_travel_pdf,
    "tavily_search_results_json": tools[1],
    "wikipedia": tools[2]
}

# --- 5. INITIALIZE LLM ---
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest", google_api_key=GEMINI_API_KEY, temperature=temp).bind_tools(tools)

# --- 6. CHAT UI ---
st.title("✈️ AI Travel Concierge")

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage): st.chat_message("user").write(msg.content)
    if isinstance(msg, AIMessage) and msg.content: st.chat_message("assistant").write(msg.content)

if user_input := st.chat_input("Ask something..."):
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        try:
            # Step 1: Call LLM
            response = llm.invoke(st.session_state.messages)
            
            # Step 2: If LLM wants to use a tool
            if response.tool_calls:
                st.session_state.messages.append(response)
                for tool_call in response.tool_calls:
                    name = tool_call["name"]
                    args = tool_call["args"]
                    # Get the tool from our map
                    output = tool_map[name].invoke(args)
                    st.session_state.messages.append(ToolMessage(content=str(output), tool_call_id=tool_call["id"]))
                
                # Step 3: Get final answer after tool results
                response = llm.invoke(st.session_state.messages)
            
            st.write(response.content)
            st.session_state.messages.append(response)

        except Exception as e:
            st.error(f"Error: {str(e)}")
