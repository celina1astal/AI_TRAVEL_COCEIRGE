import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="✈️ AI Travel Concierge", layout="wide")

# --- 2. SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a professional Travel Concierge. Always cite [Document Source] for PDF info.")
    ]

# --- 3. SIDEBAR & THEME LOGIC ---
with st.sidebar:
    st.title("⚙️ Customization")
    temp = st.slider("Temperature", 0.0, 1.0, 0.4)
    theme_choice = st.selectbox(
        "🎨 Select UI Theme", 
        ["Corporate Blue", "Nature Green", "Deep Sea", "Sunset Orange"]
    )

    # Map themes to Hex Codes
    theme_colors = {
        "Corporate Blue": {"primary": "#007BFF", "hover": "#0056b3"},
        "Nature Green": {"primary": "#28a745", "hover": "#218838"},
        "Deep Sea": {"primary": "#17a2b8", "hover": "#117a8b"},
        "Sunset Orange": {"primary": "#fd7e14", "hover": "#d35400"}
    }

    selected_color = theme_colors[theme_choice]["primary"]
    hover_color = theme_colors[theme_choice]["hover"]

    # --- EXPORT LOGIC (Fixed Indentation) ---
    def export_chat():
        chat_str = "AI TRAVEL CONCIERGE LOG\n" + "="*30 + "\n"
        for msg in st.session_state.messages:
            if not isinstance(msg, SystemMessage):
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                chat_str += f"{role}: {msg.content}\n\n"
        return chat_str

    st.download_button(
        label="📥 Download Chat Log",
        data=export_chat(),
        file_name="travel_agent_log.txt",
        mime="text/plain"
    )

# --- 4. DYNAMIC CSS ---
st.markdown(f"""
    <style>
    div.stButton > button {{
        background-color: {selected_color} !important;
        color: white !important;
        border-radius: 8px !important;
        transition: 0.3s !important;
        width: 100% !important;
    }}
    div.stButton > button:hover {{
        background-color: {hover_color} !important;
        border: 1px solid white !important;
    }}
    .stDownloadButton > button {{
        background-color: {selected_color} !important;
        color: white !important;
        width: 100% !important;
    }}
    </style>
""", unsafe_allow_html=True)

# --- 5. API KEY VALIDATION ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
except Exception:
    st.error("Missing API Keys! Check .streamlit/secrets.toml")
    st.stop()

# --- 6. TOOLS ---
@tool
def search_travel_pdf(query: str):
    """Searches the local travel manual and flight itineraries for specific details."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
        vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = vector_db.similarity_search(query, k=3)
        context = "Information found in your local documents:\n"
        for i, d in enumerate(docs):
            context += f"\n[Document Source {i+1}]: {d.page_content}\n"
        return context
    except Exception as e:
        return f"Error accessing PDF database: {str(e)}"

web_search = TavilySearchResults(tavily_api_key=TAVILY_API_KEY)
wiki_search = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

tools = [search_travel_pdf, web_search, wiki_search]
tool_map = {
    "search_travel_pdf": search_travel_pdf,
    "tavily_search_results_json": web_search,
    "wikipedia": wiki_search
}

# --- 7. INITIALIZE LLM (Hardcoded Llama for stability) ---
llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    api_key=GROQ_API_KEY,
    temperature=temp,
    max_retries=3
).bind_tools(tools)

# --- 8. UI DISPLAY ---
st.title("✈️ AI Travel Concierge")

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        st.chat_message("assistant").write(msg.content)

# --- 9. AGENTIC LOOP ---
if user_input := st.chat_input("Ask about your trip..."):
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        try:
            response = llm.invoke(st.session_state.messages)

            if response.tool_calls:
                st.session_state.messages.append(response)
                for tool_call in response.tool_calls:
                    t_name = tool_call["name"]
                    t_args = tool_call["args"]

                    with st.status(f"Acting as Agent: {t_name}...", expanded=False) as status:
                        try:
                            result = tool_map[t_name].invoke(t_args)
                            status.update(label=f"Completed {t_name}", state="complete")
                        except Exception as tool_err:
                            result = f"Technical Failure in {t_name}: {str(tool_err)}"
                            status.update(label=f"Error in {t_name}", state="error")

                        st.session_state.messages.append(
                            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                        )

                final_response = llm.invoke(st.session_state.messages)
                st.write(final_response.content)
                st.session_state.messages.append(final_response)
            else:
                st.write(response.content)
                st.session_state.messages.append(response)

        except Exception as e:
            st.error("I encountered a connection error. This is usually due to API Rate Limits.")
            print(f"DEV LOG: {e}")
