import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS  # Swapped from Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# FIXED 2026 IMPORTS (Note the _classic)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage, AIMessage

# --- CONFIG & KEYS ---
st.set_page_config(page_title="Travel AI Concierge", page_icon="✈️")
st.title("✈️ Travel AI Concierge")

# Streamlit Cloud reads these from Advanced Settings > Secrets
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

# --- DATA PROCESSING (RAG) ---
@st.cache_resource
def load_knowledge_base():
    if not os.path.exists("travel_sample.pdf"):
        st.error("Please upload 'travel_sample.pdf' to your GitHub repo!")
        return None
    
    loader = PyPDFLoader("travel_sample.pdf")
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    docs = splitter.split_documents(pages)
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", 
        google_api_key=GEMINI_API_KEY
    )
    # Using FAISS instead of Chroma to fix the Python 3.14 error
    return FAISS.from_documents(docs, embeddings)

vector_db = load_knowledge_base()

# --- TOOLS ---
@tool
def travel_kb(query: str):
    """Search the travel PDF for specific answers about trips, hotels, or flights."""
    llm_rag = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
    
    system_prompt = (
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say you don't know.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Modern Chain Logic
    question_answer_chain = create_stuff_documents_chain(llm_rag, prompt)
    rag_chain = create_retrieval_chain(vector_db.as_retriever(), question_answer_chain)
    
    response = rag_chain.invoke({"input": query})
    return response["answer"]

tools = [travel_kb]
tool_map = {t.name: t for t in tools}

# --- LLM BINDING ---
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY).bind_tools(tools)

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content="You are a professional travel assistant.")]

# Display history
for m in st.session_state.messages:
    if isinstance(m, HumanMessage): st.chat_message("user").write(m.content)
    elif isinstance(m, AIMessage) and m.content: st.chat_message("assistant").write(m.content)

# Handle Input
if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    st.session_state.messages.append(HumanMessage(content=prompt))

    with st.chat_message("assistant"):
        ai_msg = llm.invoke(st.session_state.messages)
        st.session_state.messages.append(ai_msg)
        
        if ai_msg.tool_calls:
            for tool_call in ai_msg.tool_calls:
                res = tool_map[tool_call["name"]].invoke(tool_call["args"])
                st.session_state.messages.append(ToolMessage(content=str(res), tool_call_id=tool_call["id"]))
            
            final_res = llm.invoke(st.session_state.messages)
            st.markdown(final_res.content)
            st.session_state.messages.append(final_res)
        else:
            st.markdown(ai_msg.content)
