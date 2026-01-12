# 🤖 Personal AI Assistant - Interactive Resume Agent

A  **RAG (Retrieval-Augmented Generation)** powered chatbot that serves as an **interactive resume and professional portfolio**.  
This AI assistant can answer questions about my background, skills, projects, and experience based on my actual documents and online presence.

---

## 🚀 Features

- 📚 **Multi-Source Knowledge Base:** Integrates data from PDF resume, personal info files, portfolio website, and GitHub profile.  
- 🧠 **Intelligent Q&A:** Uses RAG architecture to provide accurate, context-aware responses.  
- 💬 **Natural Conversations:** Maintains chat history for coherent and contextual discussions.  
- 🎨 **Modern UI:** Clean, professional interface similar to ChatGPT or Claude.  
- ⚡ **One-Click Questions:** Pre-loaded example questions for quick interaction.  
- 🌐 **Real-time Web Scraping:** Automatically fetches the latest info from my online profiles.  
- 💼 **Professional Presentation:** Perfect for sharing with recruiters and potential employers.

---

## 🔗You can try it using this link: https://rohit991371-personalized-agent-app-5sqmdf.streamlit.app/


| Component | Technology |
|----------|------------|
| **Frontend** | Streamlit |
| **LLM** | Groq (openai/gpt-oss-20b) |
| **Framework** | LangChain |
| **Embeddings** | Ollama (`nomic-embed-text`) |
| **Vector Store** | ChromaDB |
| **Web Scraping** | BeautifulSoup4, Requests |

---

## 📦 Prerequisites

- 🐍 Python **3.8+**
- 🦙 [Ollama](https://ollama.com/) installed and running
- 🔑 Groq API key (free tier available)

---

## 📁 Add Your Personal Data

Place the following files in the project root:

- **`Rohit Gupta.pdf`** – Resume  
- **`personal_info.txt`** – A text file with my details (name, email, GitHub, etc.)

---

## 🧠 RAG Architecture (How It Works)

The system uses **Retrieval-Augmented Generation (RAG)** to answer queries with accurate, personalized context:

1. 📄 Split documents into manageable chunks.  
2. 🧬 Create vector embeddings using **Ollama**.  
3. 🗃️ Store embeddings in **ChromaDB** vector database.  
4. 🔍 Retrieve relevant context for each query.  
5. 🤖 Generate responses using **Groq LLM** with the retrieved context.

---

## 🔗 Multi-Source Integration

- 📑 **PDF Resume:** Parsed with `PyPDF2`  
- 📁 **Personal Info:** Loaded from `personal_info.txt`  
- 🌐 **Portfolio Website:** Scraped using `BeautifulSoup`  
- 🐙 **GitHub Profile:** Public data fetched and parsed automatically

---

## 🧵 Conversation Memory

The chatbot maintains a full conversation history using LangChain’s message memory, enabling:

- 💬 Context-aware follow-up questions  
- 🔁 Natural, human-like conversation flow  
- 📚 Reference to previous questions for deeper discussions

---

## 🤝 Contributing

Contributions are welcome! 🙌  
To contribute:

1. 🍴 Fork the repository  
2. 🌱 Create your feature branch:  
   ```bash
   git checkout -b feature/AmazingFeature
