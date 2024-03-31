from langchain_community.llms import Ollama
from langchain.agents import Tool
from langchain_community.embeddings import HuggingFaceEmbeddings,OllamaEmbeddings
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.vectorstores import Chroma,Qdrant
from langchain.prompts import ChatPromptTemplate,PromptTemplate,MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader,UnstructuredFileLoader, Docx2txtLoader
from pathlib import Path
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory,ConversationBufferMemory,ChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
import textwrap


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



def answer_question(question_text):
    chat_history = []
###########################-------------------------------Step1>Load Documents--------------------------######################    
    # Load documents from various sources
    loader = DirectoryLoader('docs',loader_cls=UnstructuredFileLoader)
    documents = loader.load()
#########################--------------------------------Step2>Split Documents--------------------------########################
    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=1000,
        chunk_overlap=100,
        length_function = len
    )
    docs = text_splitter.split_documents(documents)
##############################----------------------------------Step3>Load Model & Embedding,and store it----------------############    
    # Create Ollama model and embeddings
    #model = ChatGroq(temperature=0, groq_api_key="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", model_name="mixtral-8x7b-32768")
    model=Ollama(model="mistral")
    embedding = OllamaEmbeddings(model="nomic-embed-text")


    #url="http://localhost:6333"
    #collection_name="gpt-db"
    #qdrant=Qdrant.from_documents(
      #  docs,embedding,url=url,prefer_gpc=False,collection_name=collection_name
    #)
    
    # Create and persist Chroma vector database
    vectordb = Chroma.from_documents(documents=docs,
                                     embedding=embedding,
                                     persist_directory="db")
    vectordb.persist()
    
    # Create a retriever from the Chroma vector database
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
###########################----------------------------------Step4>Create Prompt Template----------------------------#############    
    
    template = """You are an friendly ai assistant that help users find the most relevant and accurate answers \
                  to their questions based on the documents you have access to. \
                 When answering the questions, mostly rely on the info in documents & excel file only. \
                 If u don't know the answer say i don't know.Never Try to hallucinate.. \
                 Answer the question based only on the following context:
                  {context}"""
    qa_prompt=ChatPromptTemplate.from_messages([
        ("system",template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human","{question}")
    ])

########################-------------------------------------Step5>Create Templates for Chat History Also-----------------#########
    
    history_template="""Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", history_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    contextualize_q_chain = contextualize_q_prompt | model | StrOutputParser()


    def contextualized_question(input: dict):
        if input.get("chat_history"):
            return contextualize_q_chain
        else:
            return input["question"]

    
#################-------------------------Step7>Create Q&A Chain----------------------#################################
    chain = (
        RunnablePassthrough.assign(
        context=contextualized_question | retriever | format_docs
    )
        | qa_prompt
        | model
    )
    # Invoke the chain with the provided question text
    answer = chain.invoke({'question':question_text,"chat_history": chat_history})
    chat_history.extend([HumanMessage(content=question_text), answer])
    print(chat_history)
    #answer_text = answer.content
    result=textwrap.fill(answer,width=80)
    print(result)
    return result