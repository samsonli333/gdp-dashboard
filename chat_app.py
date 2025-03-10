import streamlit as st
from langchain import HuggingFaceHub
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import tensorflow as tf
from PIL import Image
import numpy as np
from pathlib import Path
from search import load_pdf , split_text , create_vector_store,semantic_search


model = tf.keras.models.load_model('tdc.keras')

# Set up the app title
st.title("Chat App with Photo Uploads 📷🤖")

# Initialize session state for chat history and uploaded files
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []


# Initialize LangChain with Hugging Face
def initialize_llm():
    """Initialize the Hugging Face LLM via LangChain."""
    repo_id = "deepseek-ai/DeepSeek-v3"  # Replace with your desired Hugging Face model
    llm = HuggingFaceHub(repo_id=repo_id, huggingfacehub_api_token=st.secrets["HF_TOKEN"])
    memory = ConversationBufferMemory()
    conversation = ConversationChain(llm=llm, memory=memory)
    return conversation
   

# Initialize the chatbot
if "conversation" not in st.session_state:
    st.session_state.conversation = initialize_llm()

# Function to interact with the chatbot
def chat_with_bot(prompt):
    try:
        """Get a response from the chatbot."""
        response = st.session_state.conversation.predict(input=prompt)
        return response
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Sorry, I encountered an error. Please try again."



def recognize(pic):
    new_img = Image.open(pic).resize((160,160)).convert('RGB')
    new_img = np.array([new_img])
    result = model.predict(new_img)
    return ['Image','Logo'][int(result[0][0])]


# Sidebar for file uploads
with st.sidebar:
    st.header("Upload Photos")
    uploaded_files = st.file_uploader(
        "Upload photos",
        type=["png", "jpg", "jpeg","pdf"],
        accept_multiple_files=True,
        help="Upload images (PNG, JPG,PDF)"
    )
  
    if uploaded_files:
        for uploaded_file in uploaded_files:
            uploaded_file.seek(0)
            st.session_state.uploaded_files.append(uploaded_file)   
            st.session_state.messages.append({
                "role": "system",
                "content": f"Uploaded photo: {uploaded_file.name}"
            })
    

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
            st.markdown(f"📷 **{message['content']}**")



# Display uploaded photos
for uploaded_file in st.session_state.uploaded_files:
    if uploaded_file.type.startswith("image/"):
        with st.chat_message("system"):
            # st.markdown(f"📷 **Uploaded Photo: {uploaded_file.name}**")
            img = Image.open(uploaded_file)
            img.save(f'./photo/{uploaded_file.name}')
            st.image(img, caption=uploaded_file.name, use_column_width=True)
        with st.spinner("Thinking..."):
            try:
                result = recognize(f'./photo/{uploaded_file.name}')
                with st.chat_message("assistant"):
                    st.markdown(result)
            except Exception as e:
                st.error(f"Recognize Error: {e}")
    elif uploaded_file.type.startswith("application/"):
        with st.chat_message("system"):
            save_folder = './data'
            save_path = Path(save_folder, uploaded_file.name)
            with open(save_path, mode='wb') as w:
                w.write(uploaded_file.getvalue())
                # Load the PDF
            with st.spinner("Loading PDF..."):
           
                documents = load_pdf(f'{save_folder}/{uploaded_file.name}')

                # Split the text into chunks
                st.markdown("Splitting text into chunks...")
                texts = split_text(documents)
            
         
                # Create the vector store
                st.markdown("Creating vector store...")
                vector_store = create_vector_store(texts)

           
                # Perform a semantic search
                query = "What is the main topic of the document?"
                st.markdown(f"Performing semantic search for query: '{query}'")
                result = semantic_search(query, vector_store)
                if not result is  None:
                    st.markdown("Search Result:") 
                    st.markdown(result)



# Chat input for user messages
if prompt := st.chat_input("Type your message..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get bot response
    with st.spinner("Thinking..."):
        bot_response = chat_with_bot(prompt)
    
    # Add bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    
    # Display bot response
    with st.chat_message("assistant"):
        st.markdown(bot_response)