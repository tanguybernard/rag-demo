from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from typing import List, Dict
from langchain_community.document_transformers import Html2TextTransformer



# Configuration
FORMATION_URLS = [
    "https://www.m2iformation.fr/formation-java-les-fondamentaux-de-la-programmation/JAV-SE/",
    "https://www.m2iformation.fr/formation-les-bonnes-pratiques-de-la-migration-vers-le-cloud/CLOUD-MIGR/",
    "https://www.octo.academy/catalogue/formation/ddd01-ddd-domain-driven-design/",
    "https://www.octo.academy/catalogue/formation/ajava-developper-son-api-avec-java/",
    "https://www.octo.academy/catalogue/formation/az204-formation-azure-pour-les-developpeurs/",
    

]
embeddings = OllamaEmbeddings(model="mistral")
llm = Ollama(model="mistral")

# Chargement brut sans filtrage CSS
loader = WebBaseLoader(FORMATION_URLS)
docs = loader.load()

# Transformation ciblée en texte structuré
html_transformer = Html2TextTransformer()
cleaned_docs = html_transformer.transform_documents(
    docs,
    selectors=[".content",".container"]  # Cibles prioritaires
)

# Découpage et indexation
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,  # Réduire la taille des chunks
    chunk_overlap=100,
    separators=["\n\n", 
                "Objectifs", #Octo
                "Objectifs de formation", # m2i
                "Publics concerné", #Zenika
                "Public cible"#Octo
                ]  # Points de rupture sémantiques
)
splits = text_splitter.split_documents(cleaned_docs)


# Embeddings : Transforme le texte en vecteurs avec le modèle Mistral via Ollama
vectorstore = FAISS.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# La déduplication permet d'eliminer les doublons :
# Les requêtes sur plusieurs critères (ex: "Prérequis", "Public visé") peuvent renvoyer des chunks identiques depuis différentes sources. La déduplication évite de surpondérer des informations répétitives.
def deduplicate_docs(docs: List) -> List:
    """Déduplication basée sur le contenu et les métadonnées"""
    seen = set()
    unique = []
    for doc in docs:
        identifier = (doc.page_content, frozenset(doc.metadata.items()))
        if identifier not in seen:
            seen.add(identifier)
            unique.append(doc)
    return unique

def evaluer_formations(profil: str) -> Dict:
    """Évaluation des formations par rapport au profil"""
    # Recherche des informations pertinentes
    criteres = [
        "Public visé", "Prérequis",
        "Objectifs",
    ]
    docs_pertinents = []
    for critere in criteres:
        docs_pertinents += retriever.invoke(f"{critere} {profil}")
    
    # Déduplication
    unique_docs = deduplicate_docs(docs_pertinents)
    
    # Génération de l'analyse
    context = "\n\n".join([doc.page_content for doc in unique_docs])
    prompt = f"""
    Analysez l'adéquation des formations avec ce profil :
    {profil}

    Critères d'évaluation :
    1. Adéquation au public visé
    2. Respect des prérequis


    Contexte :
    {context}

    Fournissez un top 3 des formations pour ce profil.
    """
    
    reponse = llm.invoke(prompt)
    
    return {
        "analyse": reponse,
        "sources": list(set(doc.metadata["source"] for doc in unique_docs))
    }

# Interface utilisateur
print("Évaluation personnalisée de formations")
profil_utilisateur = input("Décrivez votre profil (ex: 'Développeur Java, 3 ans d'expérience agile, besoin de certification PSM I') : ")

resultat = evaluer_formations(profil_utilisateur)

print("\nRésultats d'analyse :")
print(resultat["analyse"])
print("\nSources utilisées :")
for source in resultat["sources"]:
    print(f"- {source}")
