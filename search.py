from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# Step 1: Load the PDF file
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

# Step 2: Split the text into chunks
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Adjust chunk size as needed
        chunk_overlap=200,  # Overlap to maintain context
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.split_documents(documents)
    return texts

# Step 3: Create embeddings and store them in a vector store
def create_vector_store(texts):
    # Use HuggingFace embeddings (latest model)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store

# Step 4: Perform semantic search
def semantic_search(query, vector_store):
    # Load a Hugging Face QA model
    model_name = "deepset/roberta-base-squad2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

    # Retrieve relevant chunks from the vector store
    relevant_chunks = vector_store.similarity_search(query, k=3)
    context = " ".join([chunk.page_content for chunk in relevant_chunks])

    # Use the QA pipeline to answer the question
    result = qa_pipeline(question=query, context=context)
    return result["answer"]

# Main function
def main():
    # Path to your PDF file
    pdf_file_path = "./resume11-2.pdf"

    # Load the PDF
    print("Loading PDF...")
    documents = load_pdf(pdf_file_path)

    # Split the text into chunks
    print("Splitting text into chunks...")
    texts = split_text(documents)

    # Create the vector store
    print("Creating vector store...")
    vector_store = create_vector_store(texts)

    # Perform a semantic search
    query = "What is the main topic of the document?"
    print(f"Performing semantic search for query: '{query}'")
    result = semantic_search(query, vector_store)

    print("\nSearch Result:")
    print(result)

if __name__ == "__main__":
    main()