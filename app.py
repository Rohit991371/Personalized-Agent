import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import uuid
import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import time
from pathlib import Path

load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="Personal AI Assistant - Interactive Resume",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for Modern UI ---
st.markdown("""
<style>
    /* Hide Streamlit elements */
    .stApp > header {
        background-color: transparent;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp > div:first-child {
        margin-top: -80px;
    }
    
    /* Main container styling */
    .main-container {
        max-width: 1000px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    /* Header styling */
    .header-container {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .header-title {
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-subtitle {
        font-size: 1.3rem;
        margin-top: 1rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    .header-description {
        font-size: 1.1rem;
        margin-top: 1.5rem;
        opacity: 0.8;
        max-width: 100%;
        margin-left: auto;
        margin-right: auto;
        line-height: 1.6;
    }
    
    /* Status indicators */
    .status-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border: 1px solid #e1e8ed;
    }
    
    .status-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
    }
    
    .status-item {
        display: flex;
        align-items: center;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 10px;
        border-left: 4px solid #28a745;
    }
    
    .status-item.error {
        border-left-color: #dc3545;
    }
    
    .status-icon {
        font-size: 1.5rem;
        margin-right: 1rem;
    }
    
    .status-text {
        font-weight: 500;
        font-size: 0.95rem;
        color: #333;
    }
    
    /* Chat container styling */
    .chat-container {
        background: white;
        color: #333;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border: 1px solid #e1e8ed;
        margin-bottom: 2rem;
    }
    
    .chat-title {
        text-align: center;
        font-size: 2rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 1rem;
    }
    
    .chat-description {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    
    /* Example questions styling */
    .example-questions {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 15px;
        padding: 2rem;
        margin-bottom: 2rem;
        color: white;
    }
    
    .example-questions h3 {
        text-align: center;
        margin-bottom: 1.5rem;
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    .questions-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1rem;
        text-align: center;
    }
    
    /* Style example question buttons */
    div[data-testid="column"] > div > div > button[key^="example_q_"] {
        background: rgba(255,255,255,0.2) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
        border-radius: 25px !important;
        padding: 0.8rem 1.2rem !important;
        margin-bottom: 0.8rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        backdrop-filter: blur(10px) !important;
    }
    
    div[data-testid="column"] > div > div > button[key^="example_q_"]:hover {
        background: rgba(255,255,255,0.3) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2) !important;
    }

    /*===========================
    .question-tag {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        padding: 0.8rem 1.2rem;
        border-radius: 25px;
        margin: 0.3rem;
        font-size: 0.95rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.3);
        backdrop-filter: blur(10px);
    }
    
    .question-tag:hover {
        background: rgba(255,255,255,0.3);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    ==================================*/
    
    /* Features grid styling */
    .features-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 2rem;
        margin: 2rem 0;
    }
    
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border: 1px solid #e1e8ed;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .feature-card h3 {
        color: #333;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .feature-card p {
        color: #666;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Error/Loading states */
    .error-container {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .loading-container {
        background: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    
    /*=========================== commented out old chat styles
    /* Chat message styling */
    .stChatMessage {
        border-radius: 15px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    =========================*/
    
    /* Push chat input to bottom */
    .stChatFloatingInputContainer {
        position: sticky;
        bottom: 0;
        background: white;
        padding: 1rem 0;
        z-index: 999;
    }
    
    /* Add space for footer */
    .main .block-container {
        padding-bottom: 1rem;
    }
    
    /* Chat message styling */
    .stChatMessage {
        border-radius: 15px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 25px;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border: none;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 15px;
        border: 2px solid #e1e8ed;
        padding: 1rem;
        font-size: 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- Configuration ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY") 
PORTFOLIO_URL = "https://www.rohitgupta1604.com.np/" 
GITHUB_URL = "https://github.com/Rohit991371"

def get_embeddings():
    try:
        return OllamaEmbeddings(model="nomic-embed-text")
    except Exception as e:
        st.error(f"Error initializing embeddings: {e}")
        st.info("Please make sure Ollama is running with the nomic-embed-text model installed.")
        st.stop()

def get_llm():
    try:
        if GROQ_API_KEY == "your_groq_api_key_here" or not GROQ_API_KEY:
            st.error("Please set your GROQ_API_KEY in the secrets or environment variables")
            st.info("Get your free API key from: https://console.groq.com/keys")
            st.stop()
        
        return ChatGroq(
            groq_api_key=GROQ_API_KEY, 
            model_name="openai/gpt-oss-20b",  # Updated to available model
            temperature=0.1,
            max_tokens=1000
        )
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        st.stop()

# --- Session State Initialization ---
session_id = "personal_assistant_session"

if "store" not in st.session_state:
    st.session_state.store = {}
    
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    
if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "agent_ready" not in st.session_state:
    st.session_state.agent_ready = False
    
if "initial_load" not in st.session_state:
    st.session_state.initial_load = True

if "sources_status" not in st.session_state:
    st.session_state.sources_status = []
    
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# --- Header ---
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🤖 Personal AI Assistant</h1>
    <p class="header-subtitle">Interactive Resume & Professional Profile</p>
    <p class="header-description">
        I'm an AI-powered assistant that knows everything about Rohit's professional background, skills, projects, and experience. Ask me anything you'd like to know!
    </p>
</div>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def scrape_website(url):
    """Scrape content from a website."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text[:15000]  # Limit to first 15k characters
    
    except Exception as e:
        return None

def load_all_documents():
    """Load all available documents"""
    documents = []
    sources_loaded = []
    
    # Load local files
    data_dir = Path(".")
    
    # Load resume
    resume_files = list(data_dir.glob("*.pdf")) + list(data_dir.glob("*resume*.pdf")) + list(data_dir.glob("Rohit Gupta.pdf"))
    if resume_files:
        try:
            loader = PyPDFLoader(str(resume_files[0]))
            docs = loader.load()
            for doc in docs:
                doc.metadata['source'] = 'resume'
            documents.extend(docs)
            sources_loaded.append("✅ Resume (PDF)")
        except Exception as e:
            sources_loaded.append(f"❌ Resume: {str(e)}")
    else:
        sources_loaded.append("❌ Resume: No PDF files found")
    
    # Load personal info
    info_files = list(data_dir.glob("personal_info.txt")) + list(data_dir.glob("info.txt"))
    if info_files:
        try:
            with open(info_files[0], 'r', encoding='utf-8') as f:
                content = f.read()
            from langchain.schema import Document
            doc = Document(page_content=content, metadata={'source': 'personal_info'})
            documents.append(doc)
            sources_loaded.append("✅ Personal Information (TXT)")
        except Exception as e:
            sources_loaded.append(f"❌ Personal Info: {str(e)}")
    else:
        sources_loaded.append("❌ Personal Info: No personal_info.txt found")
    
    # Load portfolio website
    if PORTFOLIO_URL:
        content = scrape_website(PORTFOLIO_URL)
        if content:
            from langchain.schema import Document
            doc = Document(page_content=content, metadata={'source': 'portfolio_website', 'url': PORTFOLIO_URL})
            documents.append(doc)
            sources_loaded.append("✅ Portfolio Website")
        else:
            sources_loaded.append("❌ Portfolio Website (failed to scrape)")
    
    # Load GitHub profile
    if GITHUB_URL:
        content = scrape_website(GITHUB_URL)
        if content:
            from langchain.schema import Document
            doc = Document(page_content=content, metadata={'source': 'github_profile', 'url': GITHUB_URL})
            documents.append(doc)
            sources_loaded.append("✅ GitHub Profile")
        else:
            sources_loaded.append("❌ GitHub Profile (failed to scrape)")
    
    return documents, sources_loaded

# --- Initialize Agent on First load ---
def initialize_agent():
    """Initialize the agent with all data sources."""
    try:
        with st.spinner("🔄 Initializing AI Assistant..."):
            embeddings = get_embeddings()
            llm = get_llm()
            documents, sources = load_all_documents()
            
            if documents:
                # Text splitting
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", " ", ""]
                )
                texts = text_splitter.split_documents(documents)
                
                # Create vector store
                vectorstore = Chroma.from_documents(texts, embeddings, collection_name=f"personal_assistant_{int(time.time())}")
                retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

                st.session_state.vectorstore = vectorstore
                st.session_state.retriever = retriever
                st.session_state.sources_status = sources
                
                st.session_state.agent_ready = True
                return True, sources
            else:
                st.session_state.agent_ready = False
                st.session_state.sources_status = sources
                return False, sources
        
    except Exception as e:
        st.session_state.agent_ready = False
        error_msg = f"❌ Initialization error: {str(e)}"
        st.session_state.sources_status = [error_msg]
        return False, [error_msg]

# Initialize agent if needed
if st.session_state.initial_load or not st.session_state.agent_ready:
    success, sources = initialize_agent()
    st.session_state.initial_load = False



for source in st.session_state.sources_status:
    is_success = source.startswith("✅")
    status_class = "" if is_success else "error"
    icon = "✅" if is_success else "❌"


st.markdown("</div></div>", unsafe_allow_html=True)

# --- RAG Chain Setup and Chat Interface ---
if st.session_state.agent_ready and st.session_state.retriever:
    retriever = st.session_state.retriever
    llm = get_llm()
    
    # Custom system prompt for personal assistant
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a personal AI assistant representing a professional candidate whose name is Rohit Gupta. 
        If user 'whom are you' or 'who created you' or similar, respond with 'I am an AI assistant created by Rohit Gupta to answer questions about his professional background, skills, experience, and projects based on the documents provided to me.'
        Given the chat history and the latest user question, formulate a standalone question 
        that captures the full context needed to answer about the candidate's background, 
        skills, experience, or projects."""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful and professional AI assistant representing a job candidate whose name is Rohit Gupta. Rohit Gupta created you to answer questions about his professional background, skills, experience, and projects based on the documents provided to you. 
        if user 'whom are you' or 'who created you' or similar, respond with 'I am an AI assistant created by Rohit Gupta to answer questions about his professional background, skills, experience, and projects based on the documents provided to me.'
        
        Use the provided context to answer questions about:
        
        - Professional background and work experience
        - Technical skills, programming languages, and certifications  
        - Projects, achievements, and accomplishments
        - Contact information, social media, and portfolio links
        - Educational background and qualifications
        - Any other relevant professional information
        
        IMPORTANT GUIDELINES:
        - Don't provide information about Rohit's private github repositories or inaccessible links. Only give information about public repositories.
        - Only answer based on the provided context from the candidate's documents
        - Be conversational, professional, and enthusiastic about the candidate's abilities
        - If information is not available in the context, politely say "I don't have that specific information in my knowledge base"
        - When mentioning contact details, links, or URLs, include them exactly as they appear
        - Highlight key achievements and strengths when relevant
        - Be specific about technologies, tools, and skills when mentioned
        - Always maintain a positive and confident tone about the candidate
        - If asked about salary, availability, or interview scheduling, direct them to contact directly
        - Format your responses using plain text and Markdown syntax only
        - Use **bold** for emphasis instead of HTML tags
        - Use line breaks (double space) instead of <br> tags
        - Do NOT use HTML tags like <br>, <div>, <span>, etc.

        
        Context: {context}"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
    
    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    
    # Display chat history
    history = get_session_history(session_id)
    
    
    
        # =======================================================
    # Show example questions if no chat history
    if not history.messages:
        st.markdown("""
        <div class="example-questions">
            <h3>💡 Here are some questions you can ask:</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Define example questions
        example_questions = [
            "What are your key technical skills?",
            "Tell me about your recent projects",
            "What's your educational background?",
            "How can I contact you?",
            "What certifications do you have?",
            "Show me your GitHub profile",
            "What programming languages do you know?",
            "Describe your work experience"
        ]
        
        
    # Create columns for buttons
        cols = st.columns(2)
        for idx, question in enumerate(example_questions):
            with cols[idx % 2]:
                if st.button(question, key=f"example_q_{idx}", use_container_width=True):
                    st.session_state.pending_question = question
                    st.rerun()
                    

# =======================================================


    # Display chat history
    for msg in history.messages:
        if msg.type == "human":
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif msg.type == "ai":
            with st.chat_message("assistant"):
                st.markdown(msg.content)
    
    
    #======================================================
     # Process pending question from example buttons
    if st.session_state.pending_question:
        user_input = st.session_state.pending_question
        st.session_state.pending_question = None
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    response = conversational_rag_chain.invoke(
                        {"input": user_input},
                        config={"configurable": {"session_id": session_id}}
                    )
                    st.markdown(response['answer'], unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"I encountered an error processing your question: {str(e)}")
                    st.info("Please try asking your question again or rephrase it.")
        
        st.rerun()
    #======================================================
    
    # Chat input
    user_input = st.chat_input("Ask me anything about my background, skills, projects, or experience...")
    
    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    response = conversational_rag_chain.invoke(
                        {"input": user_input},
                        config={"configurable": {"session_id": session_id}}
                    )
                    st.markdown(response['answer'], unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"I encountered an error processing your question: {str(e)}")
                    st.info("Please try asking your question again or rephrase it.")
        
        st.rerun()

else:
    # Show features and status when agent is not ready
    if st.session_state.sources_status:
        error_sources = [s for s in st.session_state.sources_status if s.startswith("❌")]
        if error_sources:
            st.markdown('<div class="error-container">', unsafe_allow_html=True)
            st.error("⚠️ **Setup Required**")
            st.markdown("The following issues need to be resolved:")
            for error in error_sources:
                st.markdown(f"• {error.replace('❌ ', '')}")
            
            st.markdown("**Quick Setup Instructions:**")
            st.markdown("1. Add your resume as a PDF file in the project folder")
            st.markdown("2. Create a `personal_info.txt` file with your details")
            st.markdown("3. Make sure Ollama is running with `nomic-embed-text` model")
            st.markdown("4. Set your GROQ_API_KEY in secrets or environment variables")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Show features
    st.markdown("""
    <div class="features-grid">
        <div class="feature-card">
            <span class="feature-icon">🧠</span>
            <h3>Intelligent Q&A</h3>
            <p>Ask me anything about skills, experience, projects, and get detailed, accurate answers based on real data.</p>
        </div>
        
        <div class="feature-card">
            <span class="feature-icon">📄</span>
            <h3>Comprehensive Knowledge</h3>
            <p>I know everything from the resume, personal info, portfolio website, and GitHub profile.</p>
        </div>
        
        <div class="feature-card">
            <span class="feature-icon">💬</span>
            <h3>Natural Conversation</h3>
            <p>Maintain context across our conversation for a natural, flowing discussion about qualifications.</p>
        </div>
        
        <div class="feature-card">
            <span class="feature-icon">🔗</span>
            <h3>Direct Links & Contact</h3>
            <p>Get exact contact information, project links, social media profiles, and portfolio URLs.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Refresh button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Try Loading Data Again", type="primary"):
            st.session_state.initial_load = True
            st.rerun()
