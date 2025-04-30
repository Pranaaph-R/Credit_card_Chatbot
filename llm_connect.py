from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

#Gorq api configuration
llm = ChatGroq(
    model_name="llama3-8b-8192",
    temperature=0.3,
    api_key= api_key
    
)

embedding_fn = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

#loading the vectordb
db = Chroma(persist_directory="./vectorDB_index", embedding_function=embedding_fn)

collection = db._collection 

data = collection.get(include=['metadatas'])

all_metadatas = data['metadatas']

card_names = sorted({m['card_name'] for m in all_metadatas if 'card_name' in m})

st.title("💳 QA Credit Card Chatbot")

selected_card = st.selectbox("Select your credit card:", card_names)

#Getting User input
query = st.text_input("Ask your question:")

if query:
    # query = "Do i need to pay annual membership fee for Delta SkyMiles Platinum American Express Card?"
    docs = db.similarity_search(query, k=10,filter={"card_name": selected_card})

    # print (docs)

    context = "\n\n".join([doc.page_content for doc in docs])

    messages = [
        SystemMessage(content="You are a helpful assistant. Answer only using the context below. If you don’t know, say 'I don’t know.'"),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
    ]


    response = llm(messages)
    print("Answer:", response.content)
    st.subheader("💬 Answer")
    st.write(response.content)
