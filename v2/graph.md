

graph TD
    A[URLs des formations] --> B[WebBaseLoader]
    B --> C[Documents bruts]
    C --> D[Text Splitter]
    D --> E[Chunks textuels]
    E --> F[Embeddings]
    F --> G[Base vectorielle FAISS]
    G --> H[Requête utilisateur]
    H --> I[Documents pertinents]
    I --> J[LLM Mistral]
    J --> K[Analyse structurée]
