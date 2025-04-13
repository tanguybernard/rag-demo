from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from typing import List, Dict
from langchain_community.document_transformers import Html2TextTransformer
from langchain_core.prompts import ChatPromptTemplate


# Configuration
FORMATION_URLS = [
    "https://www.octo.academy/catalogue/formation/ddd01-ddd-domain-driven-design/",
    "https://www.octo.academy/catalogue/formation/ajava-developper-son-api-avec-java/",
    "https://www.octo.academy/catalogue/formation/az204-formation-azure-pour-les-developpeurs/",
]

# Initialisation des modèles
embeddings = OllamaEmbeddings(model="mistral")
llm = ChatOllama(model="mistral")

# Chargement des documents
loader = WebBaseLoader(FORMATION_URLS)
docs = loader.load()

# Transformation HTML en texte ciblé
html_transformer = Html2TextTransformer()
cleaned_docs = html_transformer.transform_documents(
    docs,
    selectors=[
        "main",  # Cible la section principale
        "container",  # Cible la section principale
        "h1", "h2", "h3",  # Titres
        "#mainContent > section.details > div > div > div.subsection.public-cible"
    ]
)

# Découpage 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  
    chunk_overlap=200,
    separators=[
        "\n\n",
        "Objectifs",
        "Public cible", 
        "Prérequis",
        "Modalités pédagogiques"
    ]
)
splits = text_splitter.split_documents(cleaned_docs)

# Création de la base vectorielle
vectorstore = FAISS.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 10,  # Augmenter le nombre de documents récupérés
        "search_type": "similarity",
        "score_threshold": 0.4
    }
)

# Déduplication des résultats
def deduplicate_docs(docs: List) -> List:
    seen = set()
    unique = []
    for doc in docs:
        identifier = (doc.page_content, frozenset(doc.metadata.items()))
        if identifier not in seen:
            seen.add(identifier)
            unique.append(doc)
    return unique

# Évaluation ciblée architecte
def evaluer_formations(profil: str) -> Dict:
    
    docs_pertinents = retriever.invoke(
        f"{profil} Public cible Prérequis"  # Ajout des champs critiques
    )
    
    filtered_docs = [
        doc for doc in deduplicate_docs(docs_pertinents)
        if any(url in doc.metadata["source"] for url in FORMATION_URLS)
        and "public cible" in doc.page_content.lower()  # Attention à la casse
        and profil.lower() in doc.page_content.lower()  # Filtrage direct
    ]

    
    # Restructuration du prompt
    context = "\n---\n".join(
        f"CONTENU DE LA FORMATION:\n{doc.page_content}\n"
        for doc in filtered_docs
    )

    prompt_template = ChatPromptTemplate.from_template("""
    Analysez EXCLUSIVEMENT ces sources pour le public '{profil}' :

    {context}

    Instructions strictes :
    1. Titre exact : Extraire le H1 ou le premier titre
    2. URL : Doit correspondre à {allowed_urls}
    3. Public cible : Doit correspondre forcément au public {profil}

    Formations interdites si :
    - L'URL ne correspond pas aux sources autorisées

    Réponse EXIGÉE :
    "1. Titre exact: [titre]
    2. URL complète: [url]
    3. Public cible: "[citation]""

    Si aucune correspondance : "Aucune formation adaptée"
    """)

        

    chain = prompt_template | llm
    allowed_urls = "\n- " + "\n- ".join(FORMATION_URLS)
    reponse = chain.invoke({
        "profil": profil,
        "context": context,
        "allowed_urls": allowed_urls
    })
    
    return {
        "analyse": reponse.content,
        "sources": list(set(doc.metadata["source"] for doc in filtered_docs))
    }

while True:  # Boucle d'interaction
    print("\n💼 Veuillez décrire votre profil (ou 'q' pour quitter)")
    profil_utilisateur = input("> ")
    
    if profil_utilisateur.lower() == 'q':
        break
    
    if not profil_utilisateur.strip():
        print("❌ Veuillez entrer une description valide")
        continue
    
    print("\n🔎 Analyse en cours...")
    resultat = evaluer_formations(profil_utilisateur)


    print("\n🏆 Résultats :")
    if "Aucune formation" not in resultat["analyse"]:
        for line in resultat["analyse"].split("\n"):
            if "http" in line:
                print(f"\n🔗 {line.strip()}")
            elif line.strip().startswith(("1.", "2.", "3.")):
                print(f"✅ {line.strip()}")
    else:
        print("❌ Aucune formation adaptée trouvée")

