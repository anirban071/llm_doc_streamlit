# Chat with llm doc : LLAMA2 + LangChain + Streamlit + Google Colab using GPU

![build-kb-chatapp-shakudo-buildOnShakudo](https://github.com/anirban071/llm_doc_streamlit/assets/7694348/be4caa93-84fe-4397-9723-0ab71c11b04b)


----------------------------------------------------------------------------
### Steps:
1. creating data folder with couple of csv and pdf files
2. splitting data into chunks with some overlapping of characters between the chunks
3. formation of vertor store of faiss db with the help of bert embedding (from huggingface) and storing it locally
4. loading llama2 model from huggingface
5. user query gets converted into vector embedding using llama2 model
6. the user query embedding finds the nearest embedding from the vectorstore for betting a better context
7. query embedding + context embedding goes into llama2 model to generate answers more human like


### Model used:
1. vectorstore - faiss db (alternate options are - chroma db or painecone)
2. vectorstore embedding model - all-MiniLM-L6-v2 (alternate options are - all-MiniLM-L12-v2, google palm2 etc.)
3. llm model - chat llama2 7b parameter with 4bit quantization (alternate options are - chat llama2 7b parameter 8 or 16 bit quantization, chat llama2 13b or 70b model)


### Infrastructure used:
1. Google colab with GPU

----------------------------------------------------------------------------


## notebook execution steps - llm_langchain_streamlit_script (execution).ipynb

import os

from google.colab import drive
drive.mount('/content/drive')

parent_dir = "/content/drive/MyDrive/Colab Notebooks/llm_doc_streamlit/"

os.chdir(parent_dir)

!pip -q install -r requirements.txt

import urllib
print("Password/Enpoint IP for localtunnel is:",urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip("\n"))

!streamlit run llm_doc_streamlit_app_v1.py &>/content/logs.txt & npx localtunnel --port 8501 & curl ipv4.icanhazip.com


![image](https://github.com/anirban071/llm_doc_streamlit/assets/7694348/495ec269-14a9-4982-8ccd-d16b7f11e3c0)




