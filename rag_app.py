import os
from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(page_title="Standard RAG App", layout="wide")
st.title("RAG Chatbot (FAISS + Gemini Embeddings + Groq)")


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
        chunk_size=300,
        chunk_overlap=30
    )
    return splitter.split_documents(documents)



def build_vectorstore(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    texts = [doc.page_content for doc in chunks]

    all_embeddings = []
    batch_size = 20

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        emb = embeddings.embed_documents(batch)
        all_embeddings.extend(emb)

    text_embedding_pairs = list(zip(texts, all_embeddings))

    return FAISS.from_embeddings(text_embedding_pairs,embedding=embeddings)
    # vectorstore = FAISS.from_documents(chunks, embeddings)
    # return vectorstore


@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3
    )


def build_rag_chain(vectorstore, llm):

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template("""
Answer ONLY using the context below.
If not found, say "I don't have that information in the document."

Context:
{context}

Question: {question}

Answer in 2-3 sentences.
""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def rag(query):
        docs = retriever.invoke(query)
        context = format_docs(docs)

        chain = prompt | llm | StrOutputParser()
        return chain.invoke({
            "context": context,
            "question": query
        })

    return rag


uploaded_file = st.file_uploader(
    "Upload a document",
    type=["txt", "pdf", "docx"]
)

if uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name}")

    documents = load_document(uploaded_file)
    chunks = chunk_documents(documents)

    vectorstore = build_vectorstore(chunks)
    llm = get_llm()
    rag = build_rag_chain(vectorstore, llm)

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

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = rag(query)
                except Exception as e:
                    answer = "Temporary error. Please try again."
                    st.error(e)

                st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

else:
    st.info("Upload a document to begin.")
