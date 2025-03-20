import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import ChatHuggingFace
import tensorflow as tf
from PIL import Image
import numpy as np
from pathlib import Path
from search import load_pdf, split_text , create_vector_store,semantic_search



if "sidebar" not in st.session_state:
    st.session_state.sidebar = 'auto'

model = tf.keras.models.load_model('tdc.keras')

st.set_page_config(initial_sidebar_state=st.session_state.sidebar)

# Set up the app title
st.title("Chat App with Photo Uploads 📷🤖")


# Initialize session state for chat history and uploaded files
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = ''

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_files = dict()

if "file_path" not in st.session_state:
    st.session_state.file_path = ''

if "pdf_message" not in st.session_state:
    st.session_state.pdf_message = True


# Initialize LangChain with Hugging Face
def initialize_llm():
    """Initialize the Hugging Face LLM via LangChain."""
    repo_id = "deepseek-ai/DeepSeek-v3" 
    llm = HuggingFaceHub(repo_id=repo_id, huggingfacehub_api_token=st.secrets["HF_TOKEN"])
    return llm


chat_model = ChatHuggingFace(llm=initialize_llm(),verbose=True)


def recognize(file_path):
    new_img = Image.open(file_path).resize((160,160)).convert('RGB')
    new_img = np.array([new_img])
    result = model.predict(new_img)
    return ['Image','Logo'][int(result[0][0])]


def reply_pdf(question):    
    result = semantic_search(question, st.session_state.vector_store)
    return result



# Initialize the chatbot
if "conversation" not in st.session_state:
    st.session_state.conversation = chat_model



# Sidebar for file uploads
with st.sidebar:
    with st.form("my_form",clear_on_submit=True):
        st.header("Upload Photos")
        uploaded_files = st.file_uploader(
        "Upload photos",
        type=["png", "jpg", "jpeg","pdf"],
        accept_multiple_files=False,
        help="Upload images (PNG, JPG,PDF)"
        )
        st.session_state.uploaded_files = uploaded_files
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.session_state.sidebar = "collapsed"
            st.rerun()
            
  
if uploaded_files:
    if uploaded_files.type.startswith('image/'):
        img = Image.open(uploaded_files)
        file_path = f'./photo/{uploaded_files.name}'
        img.save(file_path)
        st.session_state.file_path = file_path
    elif uploaded_files.type.startswith('application/'):
        save_folder = './data'
        save_path = Path(save_folder, uploaded_files.name)
        with open(save_path, mode='wb') as w:
            w.write(uploaded_files.getvalue())

                # Load the PDF
            documents = load_pdf(save_path)

                # Split the text into chunks
            st.markdown("Splitting text into chunks...")
            texts = split_text(documents)

                # Create the vector store
            st.markdown("Creating vector store...")
            st.session_state.vector_store = create_vector_store(texts)
            st.markdown("already Created Vector Store")
            st.session_state.messages.append({
                "role": "system",
                "content": f"Uploaded photo: {uploaded_files.name}"
                })
            uploaded_files = dict()



# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    elif message["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(message["content"])
    elif message["role"] == "system":
        with st.chat_message("system"):
            st.markdown(f"Already Imported 📷 **{message['content']}**")
st.session_state.messages = list()





# Chat input for user messages
if st.session_state.uploaded_files and st.session_state.uploaded_files.type.startswith('application/'):
    if st.session_state.pdf_message:
        with st.chat_message("ai"):
            st.write('please ask a question for the PDF you have just attached to')
            st.session_state.pdf_message = False
    if prompt := st.chat_input("Type your question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
            
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Get bot response
        with st.spinner("Thinking..."):
            bot_response = reply_pdf(prompt)
            
        # Add bot response to chat history
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            
        # Display bot response
        with st.chat_message("assistant"):
            st.markdown(bot_response)
elif st.session_state.uploaded_files and st.session_state.uploaded_files.type.startswith('image/'):
    with st.chat_message("ai"):
        st.markdown(f'This is {recognize(st.session_state.file_path)}')

