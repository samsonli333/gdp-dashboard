import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from pathlib import Path
from search import load_pdf, split_text , create_vector_store,semantic_search
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI




model = tf.keras.models.load_model('tdc.keras')

st.markdown(
    """
    <style>
    #MainMenu {
  visibility: hidden;
}
    </style>
    """,
    unsafe_allow_html=True
)

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
    # repo_id = "deepseek-ai/DeepSeek-v3" 
    # llm = HuggingFaceHub(repo_id=repo_id, huggingfacehub_api_token=st.secrets["HF_TOKEN"],model_kwargs={
    #                     "return_full_text":False,
    #                     })
    llm = init_chat_model("gpt-4o-mini", model_provider="openai",api_key=st.secrets["OPENAI_API_KEY"])
    return llm

model_chat = ChatOpenAI(model="gpt-4o",api_key=st.secrets["OPENAI_API_KEY"])


@tool
def recognize():
    """use when you need to answer about wether a photo or image is a logo or not."""
    if len(st.session_state.uploaded_files) > 0:
        file_path = [uploaded_file['file_path'] for uploaded_file in st.session_state.uploaded_files if uploaded_file['type'].startswith('image/')][0]
        new_img = Image.open(file_path).resize((160,160)).convert('RGB')
        new_img = np.array([new_img])
        result = model.predict(new_img)
        return ['Image','Logo'][int(result[0][0])]

@tool
def reply_pdf(input:str):
    """useful when you need to answer about pdf.
        use the entire question as input.
        for instance , if the question is 'what is the topic' , the input should be 'what is the topic'
        
        Args:
        input: First string
    """

    if len(st.session_state.uploaded_files) > 0:
        file_path = [uploaded_file['file_path'] for uploaded_file in st.session_state.uploaded_files if uploaded_file['type'].startswith('application/')][0]
     # Load the PDF
        documents = load_pdf(file_path)
    # Split the text into chunks
        st.markdown("Splitting text into chunks...")
        texts = split_text(documents)
                 
    # Create the vector store
        st.markdown("Creating vector store...")
        vector_store = create_vector_store(texts)
    # vector_store = 
        result = semantic_search(input, vector_store)
        return result



tools = [recognize,reply_pdf]


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "{input}"),
        # Placeholders fill up a **list** of messages
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_tool_calling_agent(model_chat, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)


# Initialize the chatbot
if "conversation" not in st.session_state:
    st.session_state.conversation = agent_executor


# Function to interact with the chatbot
def chat_with_bot(query):
    try:
        """Get a response from the chatbot."""
        response = st.session_state.conversation.invoke({"input":query})
        return response['output']
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Sorry, I encountered an error. Please try again."




# Sidebar for file uploads
with st.sidebar:
    with st.form("my_form",clear_on_submit=True):
        st.header("Upload Photos")
        uploaded_files = st.file_uploader(
        "Upload photos",
        type=["png", "jpg", "jpeg","pdf"],
        accept_multiple_files=True,
        help="Upload images (PNG, JPG,PDF)"
        )
        st.form_submit_button("Submit")
       
  
    if uploaded_files:
        for uploaded_file in uploaded_files:
            uploaded_file.seek(0)
            if uploaded_file.type.startswith('image/'):
                img = Image.open(uploaded_file)
                file_path = f'./photo/{uploaded_file.name}'
                img.save(file_path)
                st.session_state.uploaded_files.append({'type':uploaded_file.type,'file_path':file_path})
            elif uploaded_file.type.startswith('application/'):
                save_folder = './data'
                save_path = Path(save_folder, uploaded_file.name)
                with open(save_path, mode='wb') as w:
                    w.write(uploaded_file.getvalue())
                st.session_state.uploaded_files.append({'type':uploaded_file.type,'file_path':save_path})
            st.session_state.messages.append({
                "role": "system",
                "content": f"Uploaded photo: {uploaded_file.name}"
            })
        uploaded_files = []


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
st.session_state.messages = []







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
