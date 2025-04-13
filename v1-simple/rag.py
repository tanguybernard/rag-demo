from embedding import create_retriever
from prompt import create_chain

# Define the RAG application class
class RAGApplication:
    def __init__(self, retriever, rag_chain):
        self.retriever = retriever
        self.rag_chain = rag_chain
    def run(self, question):
        # Retrieve relevant documents
        documents = self.retriever.invoke(question)
        # Extract content from retrieved documents
        doc_texts = "\\n".join([doc.page_content for doc in documents])
        # Get the answer from the language model
        answer = self.rag_chain.invoke({"question": question, "documents": doc_texts})
        return answer

retriever = create_retriever()
rag_chain = create_chain()
rag_application = RAGApplication(retriever, rag_chain)
# Example usage
question = "What is drupal ?"
answer = rag_application.run(question)
print("Question:", question)
print("Answer:", answer)