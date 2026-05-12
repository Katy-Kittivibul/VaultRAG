import chromadb
from chromadb.config import Settings
chromadb.Client(Settings(anonymized_telemetry=False))

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception

load_dotenv()

VAULT_PATH = os.environ["OBSIDIAN_VAULT_PATH"]
CHROMA_PATH = "./data/processed/chroma_db"


def _collect_md_files(vault: Path) -> list[Path]:
    return [p for p in vault.rglob("*.md") if ".claude" not in p.parts]


def _extract_metadata(file_path: Path, vault: Path) -> dict:
    relative = file_path.relative_to(vault)
    parts = relative.parts
    return {
        "source": str(relative).replace("\\", "/"),
        "folder": parts[0] if len(parts) > 1 else "",
        "filename": file_path.stem,
    }


BATCH_SIZE = 50


def _is_rate_limit_error(exc: BaseException) -> bool:
    name = type(exc).__name__
    msg = str(exc)
    return "ResourceExhausted" in name or "429" in msg or "quota" in msg.lower()


@retry(
    retry=retry_if_exception(_is_rate_limit_error),
    wait=wait_fixed(60),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _embed_batch(chunks: list, embeddings: HuggingFaceEmbeddings, persist_dir: str) -> None:
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )


def ingest() -> None:
    vault = Path(VAULT_PATH)
    md_files = _collect_md_files(vault)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
    )

    all_chunks = []
    skipped = 0
    for file_path in md_files:
        try:
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs = loader.load()
        except Exception as exc:
            print(f"  [skip] {file_path.name}: {exc}")
            skipped += 1
            continue

        metadata = _extract_metadata(file_path, vault)
        for doc in docs:
            doc.metadata.update(metadata)

        all_chunks.extend(splitter.split_documents(docs))

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    batches = [all_chunks[i:i + BATCH_SIZE] for i in range(0, len(all_chunks), BATCH_SIZE)]
    total_batches = len(batches)
    for idx, batch in enumerate(batches, start=1):
        print(f"  Processing batch {idx}/{total_batches}...")
        _embed_batch(batch, embeddings, CHROMA_PATH)
        if idx < total_batches:
            time.sleep(60)

    print(f"Ingestion complete.")
    print(f"  Total files:  {len(md_files) - skipped} ({skipped} skipped)")
    print(f"  Total chunks: {len(all_chunks)}")


if __name__ == "__main__":
    ingest()
