from langchain_community.vectorstores import SKLearnVectorStore
from data_preparation import prepare_documents  # Import du traitement des documents
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


def create_retriever():



    # Préparation des données
    doc_splits = prepare_documents()

    local_embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vectorstore = Chroma.from_documents(documents=doc_splits, embedding=local_embeddings)

    return vectorstore.as_retriever(k=4)
