import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Modern 2026 Imports
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage, AIMessage

# --- PAGE CONFIG ---
st.set_page_config(page_title="Travel AI Concierge", page_icon="✈️")
st.title("✈️ Travel AI Concierge")

# Secrets from Streamlit Cloud (Advanced Settings > Secrets)
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

# --- DATA PROCESSING (RAG with FAISS) ---
@st.cache_resource
def load_knowledge_base():
    if not os.path.exists("travel_sample.pdf"):
        st.error("Error: 'travel_sample.pdf' not found in GitHub repository!")
        return None
    
    loader = PyPDFLoader("travel_sample.pdf")
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    docs = splitter.split_documents(pages)
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", 
        google_api_key=GEMINI_API_KEY
    )
    # Using FAISS for Python 3.14 compatibility
    return FAISS.from_documents(docs, embeddings)

vector_db = load_knowledge_base()

# --- TOOL DEFINITION ---
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

    # Retrieval logic
    question_answer_chain = create_stuff_documents_chain(llm_rag, prompt)
    rag_chain = create_retrieval_chain(vector_db.as_retriever(), question_answer_chain)
    
    response = rag_chain.invoke({"input": query})
    return response["answer"]

tools = [travel_kb]
tool_map = {t.name: t for t in tools}

# --- LLM BINDING ---
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY).bind_tools(tools)

# --- CHAT SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a professional travel assistant. Use the travel_kb tool to answer questions about the PDF.")
    ]

# Display history
for m in st.session_state.messages:
    if isinstance(m, HumanMessage): 
        st.chat_message("user").write(m.content)
    elif isinstance(m, AIMessage) and m.content: 
        st.chat_message("assistant").write(m.content)

# --- CHAT INPUT & AGENT LOOP ---
if user_query := st.chat_input("Ask about your trip..."):
    # 1. Immediately show the user's new message
    st.chat_message("user").write(user_query)
    st.session_state.messages.append(HumanMessage(content=user_query))

    with st.chat_message("assistant"):
        # 2. Call the LLM with the FULL history (including the new message)
        # This forces the LLM to move past the old "Goa" response
        response = llm.invoke(st.session_state.messages)
        
        if response.tool_calls:
            # Add the tool call to history so the AI knows it's working on it
            st.session_state.messages.append(response) 
            
            for tool_call in response.tool_calls:
                with st.status(f"Searching: {tool_call['args']['query']}...") as status:
                    tool_output = tool_map[tool_call["name"]].invoke(tool_call["args"])
                    status.update(label="Information found!", state="complete")
                
                # Add the result to history
                st.session_state.messages.append(
                    ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
                )
            
            # 3. Get the NEW final answer
            final_response = llm.invoke(st.session_state.messages)
            st.markdown(final_response.content)
            st.session_state.messages.append(final_response)
            
        else:
            # If no tool is needed, just show the plain response
            st.markdown(response.content)
            st.session_state.messages.append(response)
            # 4. CRITICAL: Force a rerun to clear the "input" state
            st.rerun()

   
