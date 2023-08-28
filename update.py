from langchain.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
# from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss_txt"


# Create vector database
def create_vector_db():
    loader = DirectoryLoader(DATA_PATH, glob="**/*.txt",
                             loader_cls=TextLoader)

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    # embeddings = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    db = FAISS.from_documents(texts, embeddings)
    db1 = FAISS.load_local(DB_FAISS_PATH, embeddings)
    db1.merge_from(db)
    db1.save_local(DB_FAISS_PATH)
    print(db1.docstore._dict)


if __name__ == "__main__":
    create_vector_db()
