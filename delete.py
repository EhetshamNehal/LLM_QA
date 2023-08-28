from langchain.document_loaders import DirectoryLoader, Docx2txtLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss_txt"


# Create vector database
def delete_index(document_name = 'data\\story6.txt'):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    dbdict = db.docstore._dict

    # Extract the IDs that match the condition
    matching_ids = [
        key for key, value in dbdict.items()
        if value.metadata.get('source', '') == document_name
    ]
    
    for id in matching_ids:
        db.delete([id])
        
    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    delete_index()
