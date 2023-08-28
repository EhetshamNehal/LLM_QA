from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from flask import Flask
from flask import Response, request
import json
import time
from langchain.llms import OpenAI
import openai
import os
from dotenv import load_dotenv
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

DB_FAISS_PATH = "vectorstore/db_faiss_txt"

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""


def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(
        template=custom_prompt_template, input_variables=["context", "question"]
    )
    return prompt


# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    
    # Including a compressor for more better results
    llm = OpenAI(temperature=0)
    compressor = LLMChainExtractor.from_llm(llm)
    
    # Combining compressor and retriever into a single retriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=db.as_retriever(search_type = "mmr")
    )
        
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=compression_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    
    return qa_chain


# Loading the model
def load_llm():
    # Load the locally downloaded model here
    # llm = CTransformers(
    #     model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
    #     model_type="llama",
    #     max_new_tokens = 512,
    #     temperature = 0.2
    # )
    llm = OpenAI(temperature=0, model="text-davinci-003")
    return llm


# QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa


# output function
def final_result(query):
    qa_result = qa_bot()

    # start_time = time.time()

    response = qa_result({"query": query})

    # end_time = time.time()
    # print("Total Time Taken : ", end_time - start_time)

    return response


@app.route("/api/qa", methods=["POST"])
def answer():
    try:
        requestParams = request.get_json()
        question = requestParams["question"]
        ans = final_result(question)
        
        print("Final answer : ", ans)

        return Response(
            response=json.dumps(
                {
                    "answer": str(ans),
                    "success": True,
                }
            ),
            status=200,
            mimetype="application/json",
        )
    except Exception as e:
        print(e)
        return Response(
            response=json.dumps(
                {
                    "answer": str(e),
                    "success": False,
                }
            ),
            status=500,
            mimetype="application/json",
        )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
