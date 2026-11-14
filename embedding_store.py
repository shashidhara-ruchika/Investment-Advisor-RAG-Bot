import chromadb
from sentence_transformers import SentenceTransformer
import streamlit as st
import logging
from config import CHROMA_PERSIST_DIR, SENTENCE_EMBEDDING_MODEL

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=True)
def init_embedding_store(qa_data):
    logger.info(f"🔧 Initializing SentenceTransformer model ({SENTENCE_EMBEDDING_MODEL})...")
    embedder = SentenceTransformer(SENTENCE_EMBEDDING_MODEL)
    logger.info("✅ SentenceTransformer model loaded")

    logger.info(f"🗄️ Connecting to ChromaDB at {CHROMA_PERSIST_DIR}...")
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    logger.info("✅ ChromaDB client connected")

    existing_collections = [
        c.name if hasattr(c, "name") else str(c)
        for c in client.list_collections()
    ]
    logger.info(f"📚 Existing collections: {existing_collections}")

    collection = None
    if "lynch_qa" in existing_collections:
        logger.info("📖 Loading existing 'lynch_qa' collection...")
        collection = client.get_collection(name="lynch_qa")
        if collection.count() < len(qa_data):
            logger.info(
                f"🔄 Collection has {collection.count()} items (< {len(qa_data)}). Re-indexing..."
            )
            client.delete_collection(name="lynch_qa")
            collection = None
        else:
            logger.info(f"✅ Existing collection loaded with {collection.count()} Q&A pairs")

    if collection is None:
        logger.info("🆕 Creating new 'lynch_qa' collection with cosine distance...")
        collection = client.create_collection(
            name="lynch_qa",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("✅ New collection created")

        logger.info(f"🔤 Encoding {len(qa_data)} questions into embeddings...")
        questions = [qa["question"] for qa in qa_data]
        embeddings = embedder.encode(
            questions,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).tolist()
        logger.info(f"✅ Generated {len(embeddings)} embeddings")

        logger.info("💾 Adding embeddings to ChromaDB collection in batches...")
        batch_size = 500
        for start in range(0, len(qa_data), batch_size):
            end = min(start + batch_size, len(qa_data))
            batch_qas = qa_data[start:end]
            batch_ids = [f"qa-{i}" for i in range(start, end)]
            batch_metadatas = [
                {
                    "question": str(item.get("question", "")),
                    "answer": str(item.get("answer", "")),
                    "source": str(item.get("source", "lynch_qa")),
                }
                for item in batch_qas
            ]
            collection.add(
                embeddings=embeddings[start:end],
                documents=[item["answer"] for item in batch_qas],
                metadatas=batch_metadatas,
                ids=batch_ids,
            )
        logger.info(f"✅ Added {len(qa_data)} Q&A pairs to ChromaDB collection")

    return embedder, collection
