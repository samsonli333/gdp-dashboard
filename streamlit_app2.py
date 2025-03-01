import streamlit as st
from langchain import HuggingFaceHub
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from PIL import Image

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
    repo_id = "deepseek-ai/DeepSeek-V3"  # Replace with your desired Hugging Face model
    llm = HuggingFaceHub(repo_id=repo_id, huggingfacehub_api_token=st.secrets["HF_TOKEN"])
    memory = ConversationBufferMemory()
    conversation = ConversationChain(llm=llm, memory=memory)
    return conversation

# Initialize the chatbot
if "conversation" not in st.session_state:
    st.session_state.conversation = initialize_llm()

# Function to interact with the chatbot
def chat_with_bot(prompt):
    """Get a response from the chatbot."""
    response = st.session_state.conversation.predict(input=prompt)
    return response

# Sidebar for file uploads
with st.sidebar:
    st.header("Upload Photos")
    uploaded_files = st.file_uploader(
        "Upload photos",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Upload images (PNG, JPG)"
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
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
            st.markdown(f"📷 **Uploaded Photo: {uploaded_file.name}**")
            img = Image.open(uploaded_file)
            img.save(f'./photo/{uploaded_file.name}')
            st.image(img, caption=uploaded_file.name, use_container_width=True)

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