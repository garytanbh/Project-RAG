import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# App title
st.title("RAG Chatbot with Web Search")

# Add clear history button
if st.button("Clear Chat History"):
    st.session_state.memory.clear()
    st.success("Chat history cleared!")

# Initialize search tool
search = DuckDuckGoSearchRun()

# Initialize session state for memory if it doesn't exist
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        k=5
    )

# File upload
uploaded_file = st.file_uploader("Upload your document", type=["pdf", "txt", "docx"])

tools = [
    Tool(
        name="Web Search",
        func=search.run,
        description="Useful for searching the internet for current information"
    )
]

if uploaded_file is not None:

    # Save uploaded file
    file_path = os.path.join("uploads", uploaded_file.name)
    os.makedirs("uploads", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif uploaded_file.name.endswith(".txt"):
        loader = TextLoader(file_path)
    elif uploaded_file.name.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        st.error("Unsupported file format")
        st.stop()

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_documents(split_docs, embeddings)
    
    # Add document retriever to tools
    retriever_tool = create_retriever_tool(
        vector_store.as_retriever(k=5),
        "Document_Search",
        "Searches the uploaded document for relevant information"
    )
    tools.append(retriever_tool)

# Initialize LLM and Agent
llm = ChatOpenAI(temperature=0.5, openai_api_key=openai_api_key)  # Increased temperature for more creative responses

# Custom prompt for more detailed responses
system_message = """You are a helpful AI assistant that can search both the internet and documents. 
When answering questions:
1. Provide detailed explanations
2. Include relevant information from uploaded PDF when user asks
3. Include relevant information from the web search when user asks
4. If using multiple sources (uploaded PDF or web search), cite them in your response
5. Aim for comprehensive but clear answers"""

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=st.session_state.memory,
    verbose=True,
    agent_kwargs={
        'system_message': system_message
    }
)

# Chat interface
st.write("### Ask a question about your document or the web")
user_input = st.text_input("Your question")

if user_input:
    try:
        response = agent.run(user_input)
        st.write("### Answer")
        st.write(response)

        # Display chat history
        with st.expander("Chat History"):
            for message in st.session_state.memory.chat_memory.messages:
                st.write(f"**{message.type}**: {message.content}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
