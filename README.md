# Chat with llm doc : LLAMA2 + LangChain + Streamlit + Google Colab using GPU (execution steps)

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




