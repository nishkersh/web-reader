import streamlit as st

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
load_dotenv()


def get_vectorstore_from_url(url):
    # get the textin document form 
    loader = WebBaseLoader(url)
    document = loader.load()

    #  split the document into cbunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    # create vectorstore from the chunks
    vector_store = Chroma.from_documents(document_chunks, FakeEmbeddings(size=1024) )

    return vector_store

def get_context_retriever_chain(vector_store):
    llm = ChatGroq()
    
    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user","Given the above conversation generate a search query to look up in order to get information relevant to the conversion")
        
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    llm = ChatGroq()

    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']


# app config
st.set_page_config(page_title="Chat With Websites",page_icon="🤖")
st.title("Chat With Websites")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot, How can I help you ? ")

    ]

# sidebar
with st.sidebar:
    st.header("settings")
    website_URL=st.text_input("Website URL")



#  disable conversion until website url is not given 

if website_URL is None or website_URL == "":
    st.info("Please enter a website URL ")
else:
    # document_chunks = get_vectorstore_from_url(website_URL)
    # vector_store = get_vectorstore_from_url(website_URL)

    # retriever_chain = get_context_retriever_chain(vector_store)

    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_URL)  
    
    
    # user input 
    user_query=st.chat_input("Type Your Message Here...")
    if user_query is not None and user_query !="":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

        # retrieved_documents = retriever_chain.invoke({
        #     "chat_history": st.session_state.chat_history,
        #     "input": user_query
        # })
        # st.write(retrieved_documents)


    #  conversion
            
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)



        # with st.chat_message("Human"):
        #     st.write(user_query)

        # with st.chat_message("AI"):
        #     st.write(response)

        # with st.sidebar:
        #     st.write(st.session_state.chat_history)

