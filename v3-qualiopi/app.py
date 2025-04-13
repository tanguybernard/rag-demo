import json
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.vectorstores import FAISS
import os

# Configuration initiale
with open("v3-qualiopi/qualiopi_criteria.json", encoding="utf-8") as f:
    criteria = {c['id']: c for c in json.load(f)}  # Transforme le json en Dictionnaire Python pour accès rapide

# Chargement des documents
def load_documents(path):
    return [
        doc for file in os.listdir(path)
        for doc in (PyPDFLoader if file.endswith(".pdf") else TextLoader)(os.path.join(path, file)).load()
    ]

# Initialisation des modèles
llm = ChatOllama(model="mistral")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Vectorisation des critères
criteria_store = FAISS.from_texts(
    [f"{c['id']}: {c['label']} - {c['description']}" for c in criteria.values()], 
    embeddings
)



# Évaluation des documents
for doc in load_documents("v3-qualiopi/docs/"):
    print(f"\n📄 Document : {doc.metadata.get('source', 'inconnu')}")
    
    # Trouver les critères pertinents
    relevant_criteria = criteria_store.similarity_search(doc.page_content, k=len(criteria))
    
    # Trier les critères par ID
    sorted_criteria = sorted(relevant_criteria, key=lambda x: int(x.page_content.split(":")[0]))
    
    for crit_vector in sorted_criteria:  
        crit_id = crit_vector.page_content.split(":")[0]
        crit = criteria.get(crit_id)
        
        if crit:  # Traitement seulement si le critère existe
            # Règle métier stricte pour le critère 2
            if crit['id'] == "2" and ("aucun" in doc.page_content.lower()):
                print(f"\n🧩 {crit['id']} - {crit['label']}\nScore: 1/5\nJustification: Absence d'objectifs pédagogiques")
                continue

            # Prompt ciblé
            response = llm.invoke(f"""
Évaluez STRICTEMENT ce document selon le critère Qualiopi suivant :

CRITÈRE [{crit['id']}] :
{crit['label']}
{crit['description']}

DOCUMENT :
{doc.page_content}

Répondez UNIQUEMENT par :
Score: X/5
Justification: [max 20 mots]
""")
            print(f"\n🧩 {crit['id']} - {crit['label']}\n{response.content.strip()}")