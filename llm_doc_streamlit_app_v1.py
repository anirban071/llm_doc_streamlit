from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import notebook_login, login
from transformers import pipeline
from langchain import HuggingFacePipeline

import streamlit as st
from streamlit_chat import message
import tempfile

import os
import sys


parent_dir = "/content/drive/MyDrive/Colab Notebooks/llm_csv_streamlit/"
DB_FAISS_PATH = parent_dir+"vectorstore/db_faiss"


#Loading embedding model
@st.cache_resource
def load_emb():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={"device": device})



#Loading llm model
@st.cache_resource
def load_llm():
    # Load the locally downloaded model here

    hf_key = "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"
    login(token=hf_key)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                          use_auth_token=True,)


    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                                device_map='auto',
                                                torch_dtype=torch.float16,
                                                use_auth_token=True,
                                                #load_in_8bit=True,
                                                load_in_4bit=True
                                                )
    pipe=pipeline("text-generation",
              model=model,
              tokenizer=tokenizer,
              torch_dtype=torch.bfloat16,
              device_map='auto',
              max_new_tokens=512,
              min_new_tokens=-1,
              top_k=30)

    llm=HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature':0.1})
    
    return llm

st.title("Chat with using Llama2 and Langchain 🦙🦜")

embeddings = load_emb()
db = FAISS.load_local(DB_FAISS_PATH, embeddings)
# db.save_local(DB_FAISS_PATH)

llm = load_llm()
memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True)
chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever(search_kwargs={'k':3}), verbose=False, memory=memory)

def conversational_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello ! Ask me anything about 🤗"]
    # st.session_state['generated'] = ["Hello ! Ask me anything about " + uploaded_file.name + " 🤗"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey ! 👋"]
    
#container for the chat history
response_container = st.container()
#container for the user's text input
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        
        user_input = st.text_input("Query:", placeholder="Talk to your data here (:", key='input')
        submit_button = st.form_submit_button(label='Send')
        
    if submit_button and user_input:
        output = conversational_chat(user_input)
        
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")



    


