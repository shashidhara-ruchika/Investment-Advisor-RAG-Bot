import logging
import streamlit as st
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from config import (
    CONVERSATIONAL_AGENT_MODEL,
    LLM_MODEL_NAME,
    MAX_NEW_TOKENS,
)

logger = logging.getLogger(__name__)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class HybridLynchAgentLLM:
    """
    Two-stage Agentic LLM Pipeline:
      1. FLAN-T5 Seq2Seq Model (`google/flan-t5-base` or `MBZUAI/LaMini-Flan-T5-248M`):
         Extracts and distills factual investment rules from the TOP_K retrieved Q&A pairs.
      2. Conversational AI Agent (`Qwen/Qwen2.5-0.5B-Instruct`):
         Synthesizes the FLAN-T5 grounded facts and retrieved Q&A context into an original,
         fluent, multi-paragraph response in Peter Lynch's voice (rather than repeating raw Q&As).
    """

    def __init__(
        self,
        flan_model_name: str = LLM_MODEL_NAME,
        chat_model_name: str = CONVERSATIONAL_AGENT_MODEL,
        use_chat_agent: bool = True,
    ):
        self.device = _get_device()
        self.flan_model_name = flan_model_name
        self.chat_model_name = chat_model_name
        self.use_chat_agent = use_chat_agent

        logger.info(f"🤖 Loading FLAN-T5 ({flan_model_name}) on {self.device}...")
        self.flan_tokenizer = AutoTokenizer.from_pretrained(flan_model_name)
        self.flan_model = AutoModelForSeq2SeqLM.from_pretrained(flan_model_name).to(self.device)
        self.flan_model.eval()
        logger.info("✅ FLAN-T5 model loaded")

        self.chat_tokenizer = None
        self.chat_model = None
        if use_chat_agent:
            logger.info(f"🎭 Loading Conversational Agent ({chat_model_name}) on {self.device}...")
            self.chat_tokenizer = AutoTokenizer.from_pretrained(chat_model_name)
            dtype = torch.float16 if self.device.type in ("cuda", "mps") else torch.float32
            self.chat_model = AutoModelForCausalLM.from_pretrained(
                chat_model_name, dtype=dtype
            ).to(self.device)
            self.chat_model.eval()
            logger.info("✅ Conversational Agent loaded")

    def distill_with_flan_t5(self, prompt: str, max_new_tokens: int = 180) -> str:
        inputs = self.flan_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.flan_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=35,
                num_beams=4,
                length_penalty=1.3,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                early_stopping=True,
            )
        return self.flan_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    def synthesize_conversational_response(
        self,
        question: str,
        flan_distilled_facts: str,
        knowledge_notes: list,
        portfolio_context: str = "",
        temperature: float = 0.6,
        max_new_tokens: int = 380,
    ) -> str:
        if not self.use_chat_agent or self.chat_model is None:
            return flan_distilled_facts

        notes_text = "\n".join(f"{i+1}. {n}" for i, n in enumerate(knowledge_notes))
        portfolio_block = (
            f"\nLive Portfolio Context:\n{portfolio_context}\n"
            if portfolio_context
            else ""
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are legendary Fidelity Magellan Fund manager Peter Lynch, author of "
                    "'One Up On Wall Street' and 'Beating the Street'. Answer the investor's "
                    "question in your warm, practical, folksy first-person voice ('I', 'my'). "
                    "IMPORTANT RULES:\n"
                    "1. DO NOT copy-paste the retrieved notes verbatim. Synthesize and explain "
                    "the concepts in your own conversational words using clear examples.\n"
                    "2. Stay 100% faithful to the exact formulas and thresholds in the retrieved notes "
                    "(for example, PEG = P/E divided by annual earnings growth rate plus dividend yield; "
                    "PEG <= 1.0 is fair/attractive, PEG <= 0.5 is a bargain, PEG > 1.5 is expensive).\n"
                    "3. Structure your answer naturally with: (a) your direct perspective, "
                    "(b) how the rule or metric works in plain English, and (c) practical advice on "
                    "what to look for and what pitfalls to avoid."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"FLAN-T5 Distilled Core Fact:\n{flan_distilled_facts}\n\n"
                    f"Retrieved Knowledge Notes from Peter Lynch's Books (TOP_K={len(knowledge_notes)}):\n"
                    f"{notes_text}\n"
                    f"{portfolio_block}\n"
                    f"Investor's Question: {question}\n\n"
                    "Please write a thoughtful, original, multi-paragraph response as Peter Lynch:"
                ),
            },
        ]

        chat_prompt = self.chat_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.chat_tokenizer([chat_prompt], return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.chat_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=max(0.2, min(float(temperature), 0.9)),
                top_p=0.9,
                repetition_penalty=1.12,
            )

        gen_ids = output_ids[0][inputs.input_ids.shape[1]:]
        text = self.chat_tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return text

    def __call__(
        self,
        prompt: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
        do_sample: bool = True,
        top_p: float = 0.9,
        top_k: int = 40,
        temperature: float = 0.7,
        **kwargs,
    ):
        text = self.distill_with_flan_t5(prompt, max_new_tokens=min(max_new_tokens, 220))
        return [{"generated_text": text}]


@st.cache_resource(show_spinner=True)
def load_llm(mode: str = "Hybrid Agent: FLAN-T5 Grounding + Conversational AI Synthesis (Recommended)"):
    if "LaMini-Flan-T5" in mode:
        return HybridLynchAgentLLM(
            flan_model_name="MBZUAI/LaMini-Flan-T5-248M",
            use_chat_agent=False,
        )
    if "Standard FLAN-T5" in mode:
        return HybridLynchAgentLLM(
            flan_model_name="google/flan-t5-base",
            use_chat_agent=False,
        )
    return HybridLynchAgentLLM(
        flan_model_name="MBZUAI/LaMini-Flan-T5-248M",
        chat_model_name=CONVERSATIONAL_AGENT_MODEL,
        use_chat_agent=True,
    )
