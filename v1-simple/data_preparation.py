from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def prepare_documents():
    # Liste des URLs à charger

    ###########@

    loader = WebBaseLoader("https://en.wikipedia.org/wiki/Drupal")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)

    return all_splits
