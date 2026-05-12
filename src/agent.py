import os

import numpy as np
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.retrievers import BM25Retriever

load_dotenv()

CHROMA_PATH = "./data/processed/chroma_db"


class SemanticCache:
    def __init__(self, threshold: float = 0.92):
        self._store: list[tuple[np.ndarray, dict]] = []
        self._threshold = threshold
        self._embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def _embed(self, text: str) -> np.ndarray:
        return np.array(self._embedder.embed_query(text))

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def get(self, question: str) -> dict | None:
        if not self._store:
            print("[CACHE MISS]")
            return None
        q_emb = self._embed(question)
        best = max(self._cosine_similarity(q_emb, emb) for emb, _ in self._store)
        if best >= self._threshold:
            print("[CACHE HIT]")
            _, result = max(self._store, key=lambda pair: self._cosine_similarity(q_emb, pair[0]))
            return result
        print("[CACHE MISS]")
        return None

    def set(self, question: str, result: dict) -> None:
        self._store.append((self._embed(question), result))


SYSTEM_PROMPT = (
    "You are an assistant that answers questions about the user's data science and ML "
    "projects, skills, and tools based on their Obsidian vault. Be concise and specific. "
    "If unsure, say so."
    "\n\n{context}"
)


def build_chain():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = Ollama(model="mistral")

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

    chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))
    return chain, SemanticCache()


def build_hybrid_chain():
    vault_path = os.environ["OBSIDIAN_VAULT_PATH"]
    loader = DirectoryLoader(
        vault_path,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        silent_errors=True,
    )
    docs = [d for d in loader.load() if ".claude" not in d.metadata.get("source", "")]

    bm25_retriever = BM25Retriever.from_documents(docs, k=3)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5],
    )

    llm = Ollama(model="mistral")
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

    chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))
    return chain, SemanticCache()


def ask(chain, question: str, cache: SemanticCache | None = None) -> dict:
    if cache is not None:
        cached = cache.get(question)
        if cached is not None:
            return cached

    result = chain.invoke({"input": question})
    output = {
        "answer": result["answer"],
        "sources": result["context"],
    }

    if cache is not None:
        cache.set(question, output)

    return output


def _print_result(result: dict) -> None:
    print(f"\nAnswer:\n{result['answer']}\n")
    if result["sources"]:
        print("Sources:")
        for doc in result["sources"]:
            m = doc.metadata
            print(f"  [{m.get('folder', '—')}] {m.get('source', 'unknown')}")
    print()


def main() -> None:
    print("Loading RAG chain...")
    chain, cache = build_chain()
    print("Ready. Type 'exit' to quit.\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question or question.lower() in ("exit", "quit"):
            break

        result = ask(chain, question, cache)
        _print_result(result)


if __name__ == "__main__":
    main()
