from langchain_community.chat_models import ChatOllama
from langchain.agents import Tool
from langchain_community.embeddings import HuggingFaceEmbeddings,OllamaEmbeddings,GPT4AllEmbeddings
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.vectorstores import Chroma,Qdrant
from langchain.prompts import ChatPromptTemplate,PromptTemplate,MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader,UnstructuredFileLoader, Docx2txtLoader
from pathlib import Path
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory,ConversationBufferMemory,ChatMessageHistory,ConversationSummaryBufferMemory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
import textwrap
from dotenv import load_dotenv
load_dotenv()

def answer_question(question, chat_history=[]):
    # Initialize Ollama model and embedding
    model = ChatOllama(model="mistral")
    embedding = GPT4AllEmbeddings()

    # Load documents from a directory
    pdf_loader = DirectoryLoader('./docs/',  glob="**/*.pdf")
    word_loader= DirectoryLoader('./docs/',  glob="**/*.docx")
    loaders=[pdf_loader,word_loader]
    documents=[]
    for i in loaders:
        documents.extend(i.load())

    # Split documents into smaller chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Create and persist Chroma vector database
    vectordb = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory="db")
    vectordb.persist()

    # Create a retriever from the Chroma vector database
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Create a prompt template
    query_wrapper_prompt = """
        You are an AI assistant that helps users find the most relevant and accurate answers to their questions. \
        The document information is below.
        -------------------------------
        {context}
        -------------------------------
        Using the document information and mostly relying on it,
        answer the query. Do not try to make up the answers on your own. Only give specific answers based on document information.
        Query: {question}
        Answer:
        """
    custom_prompt = PromptTemplate(
        template=query_wrapper_prompt,
        input_variables=["context", "question"]
    )

    # Create a Conversational Retrieval Chain
    chain = ConversationalRetrievalChain.from_llm(llm=model, chain_type='stuff',
                                                  retriever=retriever,
                                                  combine_docs_chain_kwargs={"prompt": custom_prompt})

    # Get the answer
    result = chain({'question': question, 'chat_history': chat_history})
    answer = result['answer']

    # Update chat history
    chat_history.append((question, answer))

    return answer, chat_history