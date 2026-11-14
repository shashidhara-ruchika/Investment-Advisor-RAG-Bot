import os
import pandas as pd
import streamlit as st
import logging
from config import QA_FILE
from utils.convert_to_unicode import build_and_convert_dataset

logger = logging.getLogger(__name__)


@st.cache_data(show_spinner=True)
def load_qa():
    if not os.path.exists(QA_FILE):
        logger.info(f"⚙️ {QA_FILE} not found. Running data ingestion pipeline...")
        build_and_convert_dataset("data/peter_lynch_qna.csv", QA_FILE)

    logger.info(f"📖 Loading Q&A data from {QA_FILE}...")
    df = pd.read_csv(QA_FILE).fillna("")
    logger.info(f"📊 CSV loaded with {len(df)} rows")

    qa_data = []
    for _, row in df.iterrows():
        q = str(row.get("Questions", "")).strip()
        a = str(row.get("Answers", "")).strip()
        src = str(row.get("source_file", "peter_lynch_qna_unicode.csv")).strip()
        if q and a:
            qa_data.append({
                "question": q,
                "answer": a,
                "source": src,
            })

    logger.info(f"🔄 Converted {len(qa_data)} Q&A pairs to dictionary format")
    return qa_data
