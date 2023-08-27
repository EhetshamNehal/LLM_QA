from langchain.embeddings import HuggingFaceEmbeddings
# from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.document_loaders import Docx2txtLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.document_loaders.csv_loader import CSVLoader

DATA_PATH = 'csv/'
DB_FAISS_PATH = 'vectorstore/db_faiss_csv'

# Create vector database
def create_vector_db():
    loader = DirectoryLoader(DATA_PATH,
                             glob='*.docx',
                             loader_cls=Docx2txtLoader)
    
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
    #embeddings = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    create_vector_db()

