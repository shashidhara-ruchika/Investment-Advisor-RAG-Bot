QA_FILE = "data/peter_lynch_qna_unicode.csv"
CHROMA_PERSIST_DIR = "chroma_persist"
TOP_K = 3
CANDIDATE_POOL_K = 10
LLM_MODEL_NAME = "google/flan-t5-base"
CONVERSATIONAL_AGENT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
AVAILABLE_LLM_MODES = [
    "Hybrid Agent: FLAN-T5 Grounding + Conversational AI Synthesis (Recommended)",
    "MBZUAI/LaMini-Flan-T5-248M (Instruction-Tuned FLAN-T5)",
    "google/flan-t5-base (Standard FLAN-T5)",
]
MAX_NEW_TOKENS = 512
SENTENCE_EMBEDDING_MODEL = "thenlper/gte-base"
