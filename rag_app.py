import os
from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from langchain_groq import ChatGroq

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

st.set_page_config(page_title="Hybrid RAG App (Groq + Gemini)", layout="wide")

st.title("Hybrid RAG Chatbot (Groq + Gemini Embeddings + FAISS + BM25)")
st.write("Upload a document and ask questions!")


def load_document(uploaded_file):
    file_path = f"temp_{uploaded_file.name}"

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if uploaded_file.name.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")

    elif uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(file_path)

    elif uploaded_file.name.endswith(".docx"):
        loader = Docx2txtLoader(file_path)

    else:
        st.error("Unsupported file type!")
        return []

    return loader.load()


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_documents(documents)


def rewrite_query(query, llm):
    prompt = ChatPromptTemplate.from_template("""
Rewrite the query for better retrieval.
Add synonyms and detail.
Return only the improved query.

Query: {query}
""")

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"query": query})


@st.cache_resource
def build_retrievers(chunks):

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    faiss = FAISS.from_documents(chunks, embeddings)
    faiss_retriever = faiss.as_retriever(search_kwargs={"k": 5})

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 5

    hybrid = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25],
        weights=[0.5, 0.5]
    )

    return hybrid


def build_rag_chain(retriever, llm):

    prompt = ChatPromptTemplate.from_template("""
Answer ONLY using the context below.
If not found, say "I don't have that information in the document."

Context:
{context}

Question: {question}

Answer in 2-3 sentences.
""")

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama-3.1-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3
    )


uploaded_file = st.file_uploader(
    "Upload a document",
    type=["txt", "pdf", "docx"]
)

if uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name}")

    documents = load_document(uploaded_file)
    chunks = chunk_documents(documents)

    retriever = build_retrievers(chunks)
    llm = get_llm()
    rag_chain = build_rag_chain(retriever, llm)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Ask something about your document...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        rewritten_query = rewrite_query(query, llm)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = rag_chain.invoke(rewritten_query)
                st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

else:
    st.info("Please upload a file to start chatting.")