import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.llms import Bedrock

# Process individual files
def process_files(files):
    loaders = []
    for file_path in files:
        print(f"Processing file: {file_path}")  # Debugging statement
        if file_path.endswith('.pdf'):
            print(f"Recognized as PDF file: {file_path}")
            loaders.append(PyPDFLoader(file_path))
        elif file_path.endswith('.txt'):
            print(f"Recognized as text file: {file_path}")
            loaders.append(TextLoader(file_path))
        elif file_path.endswith('.csv'):
            print(f"Recognized as CSV file: {file_path}")
            loaders.append(CSVLoader(file_path))
        else:
            print(f"Unsupported file format: {file_path}")

    if not loaders:
        raise ValueError("No valid documents found to create an index.")
    
    data_split = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=30, chunk_overlap=10)
    data_embedding = BedrockEmbeddings(
        credentials_profile_name='default',
        region_name="us-east-1",
        model_id="amazon.titan-embed-text-v1"
    )
    data_index = VectorstoreIndexCreator(
        text_splitter=data_split,
        embedding=data_embedding,
        vectorstore_cls=FAISS
    )
    db_index = data_index.from_loaders(loaders)
    return db_index

def pdeu_llm():
    llm = Bedrock(
        credentials_profile_name='default',
        region_name="us-east-1",
        model_id="amazon.titan-text-express-v1",
        model_kwargs={
            "maxTokenCount": 3072,
            "temperature": 0.7,
            "topP": 0.9
        }
    )
    return llm

def pdeu_rag_response(index, question):
    rag_llm = pdeu_llm()
    pdeu_rag_query = index.query(question=question, llm=rag_llm)
    return pdeu_rag_query
