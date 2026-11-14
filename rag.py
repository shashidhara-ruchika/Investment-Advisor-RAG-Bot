import logging
import re
from config import TOP_K, CANDIDATE_POOL_K, MAX_NEW_TOKENS

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "of", "with", "by", "about", "as",
    "into", "like", "through", "after", "over", "between", "out", "against",
    "during", "without", "before", "under", "around", "among", "what",
    "how", "why", "when", "where", "who", "which", "do", "does", "did",
    "can", "could", "should", "would", "will", "you", "your", "yours",
    "peter", "lynch", "lynchs", "his", "him", "he", "i", "my", "me",
}

FINANCIAL_DOMAIN_TERMS = {
    "peg", "pe", "p/e", "earnings", "growth", "debt", "equity", "cash",
    "free", "cashflow", "dividend", "stalwart", "stalwarts", "cyclical",
    "cyclicals", "turnaround", "turnarounds", "fast", "grower", "growers",
    "slow", "asset", "play", "tenbagger", "tenbaggers", "inventory",
    "inventories", "margin", "margins", "roe", "roa", "balance", "sheet",
    "buyback", "buybacks", "spinoff", "spinoffs", "insider", "insiders",
    "diworsification", "magellan", "fund", "portfolio", "sell", "selling",
    "buy", "buying", "retail", "restaurant", "bank", "banks", "savings",
}


def preprocess_query(query: str) -> str:
    """
    Stage 1: Preprocess the query to normalize conversational pronouns to "Peter Lynch"
    and expand financial shorthand for improved vector retrieval in ChromaDB.
    """
    replacements = [
        (r"\b(you|your|yours)\b", "Peter Lynch"),
        (r"\b(peter|lynch)\b", "Peter Lynch"),
        (r"\b(his|him|he)\b", "Peter Lynch"),
        (r"\b(this investor|this fund manager)\b", "Peter Lynch"),
    ]

    processed_query = query.strip()
    for pattern, replacement in replacements:
        processed_query = re.sub(
            pattern, replacement, processed_query, flags=re.IGNORECASE
        )

    processed_query = re.sub(
        r"(Peter Lynch[\s']*)+", "Peter Lynch ", processed_query
    ).strip()
    processed_query = re.sub(
        r"Peter Lynch\s+and\s+Peter Lynch", "Peter Lynch", processed_query
    )

    logger.info(f"📝 Original Query:  '{query}'")
    logger.info(f"🔄 Processed Query: '{processed_query}'")
    return processed_query


def _tokenize(text: str):
    tokens = re.findall(r"[a-z0-9/]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def compute_contextual_rerank_score(
    original_query: str,
    processed_query: str,
    candidate_question: str,
    candidate_answer: str,
    semantic_similarity: float,
) -> dict:
    """
    Stage 3: Contextual Re-Ranking Score computation.
    Combines:
      1. Semantic Similarity Score from ChromaDB SentenceTransformer embeddings (70% weight)
      2. Lexical / Keyword Overlap Score on Question + Answer (20% weight)
      3. Financial Domain & Methodology Concept Boost (10% weight)
    """
    query_tokens = set(_tokenize(original_query) + _tokenize(processed_query))
    q_tokens = set(_tokenize(candidate_question))
    a_tokens = set(_tokenize(candidate_answer))
    doc_tokens = q_tokens | a_tokens

    if not query_tokens:
        lexical_score = 0.0
        domain_boost = 0.0
    else:
        q_overlap = len(query_tokens & q_tokens) / max(len(query_tokens), 1)
        doc_overlap = len(query_tokens & doc_tokens) / max(len(query_tokens), 1)
        lexical_score = 0.65 * q_overlap + 0.35 * doc_overlap

        matched_domain_terms = (query_tokens & doc_tokens) & FINANCIAL_DOMAIN_TERMS
        domain_boost = min(len(matched_domain_terms) * 0.25, 1.0)

    final_score = (
        0.70 * semantic_similarity
        + 0.20 * lexical_score
        + 0.10 * domain_boost
    )

    return {
        "semantic_score": round(float(semantic_similarity), 4),
        "lexical_score": round(float(lexical_score), 4),
        "domain_boost": round(float(domain_boost), 4),
        "rerank_score": round(float(final_score), 4),
    }


def _normalize_to_first_person(text: str) -> str:
    """Convert third-person descriptions of Peter Lynch into natural first-person notes."""
    out = text.strip()
    out = re.sub(
        r"^(According to (Peter )?Lynch,\s*|In (Peter )?Lynch's view,\s*)",
        "",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(
        r"\b(Peter )?Lynch\s+(says|believes|teaches|observes|notes|warns|advises|recommends|explains|suggests|points out|emphasizes|insists|reminds investors|states|views|thinks|popularized|coined)\s+(that\s+)?",
        "I always emphasize that ",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(r"\bPeter Lynch's\b", "my", out, flags=re.IGNORECASE)
    out = re.sub(r"\bLynch's\b", "my", out, flags=re.IGNORECASE)
    out = re.sub(r"\bPeter Lynch\b", "I", out, flags=re.IGNORECASE)
    out = re.sub(r"\bLynch\b", "I", out, flags=re.IGNORECASE)
    return out


def retrieve_and_rerank(
    query: str,
    embedder,
    collection,
    top_k: int = TOP_K,
    candidate_pool_k: int = CANDIDATE_POOL_K,
    enable_reranking: bool = True,
):
    """
    Executes Stages 1-3 of the Multi-Stage RAG Pipeline:
      - Stage 1: Query Preprocessing & Normalization
      - Stage 2: ChromaDB Semantic Vector Search (over-fetching candidate pool)
      - Stage 3: Contextual Re-Ranking down to configurable TOP_K
    """
    processed_query = preprocess_query(query)

    embeddings = embedder.encode(
        [query, processed_query], normalize_embeddings=True
    )
    query_embedding = ((embeddings[0] + embeddings[1]) / 2.0).tolist()

    fetch_k = max(top_k * 4, candidate_pool_k) if enable_reranking else top_k
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=fetch_k,
        include=["metadatas", "distances", "documents"],
    )

    metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
    distances = results.get("distances", [[]])[0] if results.get("distances") else []

    candidates = []
    seen_norm = set()
    for idx, meta in enumerate(metadatas):
        q_text = meta.get("question", "")
        a_text = meta.get("answer", "")
        if not q_text or not a_text:
            continue

        norm_sig = re.sub(r"[^a-z0-9]", "", _normalize_to_first_person(a_text).lower())[:90]
        if norm_sig in seen_norm:
            continue
        seen_norm.add(norm_sig)

        dist = distances[idx] if idx < len(distances) else 0.5
        sem_sim = max(0.0, min(1.0, 1.0 - (float(dist) / 2.0 if dist > 1.0 else float(dist))))

        scores = compute_contextual_rerank_score(
            original_query=query,
            processed_query=processed_query,
            candidate_question=q_text,
            candidate_answer=a_text,
            semantic_similarity=sem_sim,
        )

        candidates.append({
            "question": q_text,
            "answer": a_text,
            "source": meta.get("source", "lynch_qa"),
            **scores,
        })

    if enable_reranking:
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    else:
        candidates.sort(key=lambda x: x["semantic_score"], reverse=True)

    top_candidates = candidates[:top_k]

    telemetry = {
        "original_query": query,
        "processed_query": processed_query,
        "candidate_pool_size": len(candidates),
        "top_k": top_k,
        "reranking_enabled": enable_reranking,
        "all_candidates": candidates,
    }
    return top_candidates, telemetry


def _safe_float(val, default: float = 0.0) -> float:
    try:
        if val is None:
            return float(default)
        f = float(val)
        if f != f or f == float("inf") or f == float("-inf"):
            return float(default)
        return f
    except Exception:
        return float(default)


def evaluate_stock_lynch_agent(symbol: str, metrics: dict) -> dict:
    """
    Conversational AI Agent emulating Peter Lynch's methodology on real-time stock metrics.
    Classifies a company into one of Peter Lynch's 6 Categories and evaluates its
    balance sheet solvency, growth rate, profitability, and valuation.
    """
    rev_growth = _safe_float(metrics.get("revenueGrowth"), 0.0)
    earn_growth = _safe_float(metrics.get("earningsGrowth"), 0.0)
    debt_to_eq = _safe_float(metrics.get("debtToEquity"), 0.0)
    curr_ratio = _safe_float(metrics.get("currentRatio"), 0.0)
    roe = _safe_float(metrics.get("returnOnEquity"), 0.0)
    profit_margin = _safe_float(metrics.get("profitMargins"), 0.0)
    free_cf = _safe_float(metrics.get("freeCashflow"), 0.0)
    total_cash = _safe_float(metrics.get("totalCash"), 0.0)
    total_debt = _safe_float(metrics.get("totalDebt"), 0.0)
    curr_price = _safe_float(metrics.get("currentPrice"), 0.0)
    target_price = _safe_float(metrics.get("targetMeanPrice"), curr_price)

    growth_rate_pct = max(earn_growth, rev_growth) * 100.0

    if earn_growth < -0.10 or profit_margin < 0.02:
        category = "Turnaround / Cyclical Candidate"
        category_desc = (
            "Earnings or margins are currently compressed. In my playbook, you must check "
            "whether debt is manageable while waiting for industry recovery or turnaround signs."
        )
    elif growth_rate_pct >= 18.0:
        category = "Fast Grower"
        category_desc = (
            "Growing above 18-20% annually—this is the land of potential tenbaggers! "
            "Watch that expansion stays profitable and the balance sheet stays clean."
        )
    elif growth_rate_pct >= 8.0:
        category = "Stalwart"
        category_desc = (
            "A steady blue-chip compounder growing 8-15% per year. Great for 30-50% gains "
            "and portfolio protection during recessions."
        )
    elif total_cash > total_debt * 1.25 and total_cash > 0:
        category = "Asset / Cash-Rich Play"
        category_desc = (
            "Sitting on a fortress net-cash position exceeding total debt, providing "
            "downside cushion and buyback/dividend firepower."
        )
    else:
        category = "Slow Grower / Mature Dividend Payer"
        category_desc = (
            "A mature enterprise growing at single-digit rates. Own it only if the dividend "
            "is well-covered by free cash flow and valuation is modest."
        )

    if total_cash > total_debt and total_cash > 0:
        short_verdict = f"Cash (${total_cash/1e9:.1f}B) exceeds Total Debt (${total_debt/1e9:.1f}B)"
        balance_sheet_verdict = (
            f"Fortress Balance Sheet: {short_verdict}. "
            "It's almost impossible to go bankrupt when cash exceeds debt!"
        )
        solvency_score = 95
    elif 0 < debt_to_eq <= 50:
        short_verdict = f"Low Debt-to-Equity ({debt_to_eq:.1f}%)"
        balance_sheet_verdict = (
            f"Healthy Balance Sheet: {short_verdict} is well within my 75% equity / 25% debt rule."
        )
        solvency_score = 82
    elif debt_to_eq <= 120:
        short_verdict = f"Moderate Debt-to-Equity ({debt_to_eq:.1f}%), Current Ratio {curr_ratio:.2f}"
        balance_sheet_verdict = (
            f"Moderate Leverage: {short_verdict}. Monitor cash flow coverage."
        )
        solvency_score = 65
    else:
        short_verdict = f"High Debt-to-Equity ({debt_to_eq:.1f}%)"
        balance_sheet_verdict = (
            f"High Leverage Alert: {short_verdict}. In my experience, heavy debt is the #1 enemy in a downturn."
        )
        solvency_score = 40

    upside_pct = ((target_price - curr_price) / curr_price * 100.0) if curr_price > 0 else 0.0

    checklist = [
        f"**Lynch Category**: {category}",
        f"**Growth Profile**: Revenue Growth {rev_growth*100:.1f}% | Earnings Growth {earn_growth*100:.1f}%",
        f"**Profitability**: ROE {roe*100:.1f}% | Net Profit Margin {profit_margin*100:.1f}%",
        f"**Balance Sheet Check**: {balance_sheet_verdict}",
        f"**Free Cash Flow**: ${free_cf/1e9:.2f}B | Price: ${curr_price:.2f} (Analyst Target: ${target_price:.2f}, {upside_pct:+.1f}%)",
    ]

    return {
        "symbol": symbol,
        "category": category,
        "category_desc": category_desc,
        "solvency_score": solvency_score,
        "short_verdict": short_verdict,
        "balance_sheet_verdict": balance_sheet_verdict,
        "upside_pct": round(upside_pct, 2),
        "checklist": checklist,
    }


def _build_portfolio_grounding(query: str, portfolio_df) -> str:
    """
    Agentic Portfolio Reasoner: Connects the user's question to real-time portfolio holdings
    even when no specific ticker was named, or performs deep-dive analysis on named tickers.
    """
    if portfolio_df is None or portfolio_df.empty:
        return ""

    q_lower = query.lower()
    query_upper_tokens = set(re.findall(r"\b[A-Z]{1,5}\b", query.upper()))
    mentioned = [sym for sym in portfolio_df.columns if sym.upper() in query_upper_tokens]

    if mentioned:
        sections = ["### 📊 Live Peter Lynch Agent Analysis on Your Mentioned Stocks"]
        for sym in mentioned[:3]:
            rep = evaluate_stock_lynch_agent(sym, portfolio_df[sym].to_dict())
            bullets = "\n".join(f"  - {line}" for line in rep["checklist"])
            sections.append(
                f"**{sym} — {rep['category']} (Solvency Score: {rep['solvency_score']}/100)**\n"
                f"{rep['category_desc']}\n{bullets}"
            )
        return "\n\n".join(sections)

    reports = [
        evaluate_stock_lynch_agent(sym, portfolio_df[sym].to_dict())
        for sym in list(portfolio_df.columns)
    ]
    if any(k in q_lower for k in ["balance sheet", "debt", "cash", "bankrupt", "solvency", "financial"]):
        fortress = [r for r in reports if r["solvency_score"] >= 85][:4]
        leveraged = [r for r in reports if r["solvency_score"] <= 50][:3]
        lines = ["### 💼 Live Portfolio Application (Balance Sheet & Solvency Scan)"]
        if fortress:
            f_str = ", ".join(f"**{r['symbol']}** ({r['short_verdict']})" for r in fortress)
            lines.append(f"- **Fortress / Low-Debt Holdings**: {f_str}")
        if leveraged:
            l_str = ", ".join(f"**{r['symbol']}** ({r['short_verdict']})" for r in leveraged)
            lines.append(f"- **High-Leverage Holdings to Watch Closely**: {l_str}")
        return "\n".join(lines)

    if any(k in q_lower for k in ["fast grower", "tenbagger", "peg", "growth", "categor", "stalwart", "turnaround", "cyclical"]):
        fast = [r["symbol"] for r in reports if "Fast Grower" in r["category"]][:5]
        stalwarts = [r["symbol"] for r in reports if "Stalwart" in r["category"]][:5]
        turnarounds = [r["symbol"] for r in reports if "Turnaround" in r["category"]][:3]
        lines = ["### 💼 Live Portfolio Mapping Across My 6 Stock Categories"]
        if fast:
            lines.append(f"- **Fast Growers in Your Portfolio (18%+ Growth — Tenbagger Hunting Ground)**: {', '.join(f'**{s}**' for s in fast)}")
        if stalwarts:
            lines.append(f"- **Stalwarts (Steady 8–15% Blue-Chip Compounders)**: {', '.join(f'**{s}**' for s in stalwarts)}")
        if turnarounds:
            lines.append(f"- **Turnaround / Cyclical Watchlist**: {', '.join(f'**{s}**' for s in turnarounds)}")
        return "\n".join(lines)

    return ""


def get_rag_answer(
    query: str,
    embedder,
    collection,
    llm,
    top_k: int = TOP_K,
    enable_reranking: bool = True,
    temperature: float = 0.6,
    portfolio_df=None,
    return_telemetry: bool = False,
):
    """
    End-to-End Multi-Stage Agentic RAG Pipeline:
      1. Query Preprocessing & Entity/Intent Analysis
      2. ChromaDB Vector Similarity Retrieval (2,577+ Q&A pairs)
      3. Contextual Re-Ranking (with configurable TOP_K=3)
      4. Two-Stage AI Generation:
         - Stage 4A: FLAN-T5 Factual Grounding & Rule Distillation
         - Stage 4B: Conversational AI Agent Synthesis + Live Portfolio Grounding
    """
    logger.info(f"🔍 Processing query: {query[:60]}... (TOP_K={top_k})")

    context_data, telemetry = retrieve_and_rerank(
        query=query,
        embedder=embedder,
        collection=collection,
        top_k=top_k,
        enable_reranking=enable_reranking,
    )

    if not context_data:
        fallback_msg = (
            "I'm sorry, I don't have enough context on that specific topic. "
            "Ask me about my six stock categories, the PEG ratio, tenbaggers, "
            "balance sheet analysis, or any stock in your portfolio!"
        )
        return (fallback_msg, [], telemetry) if return_telemetry else (fallback_msg, [])

    knowledge_notes = [
        _normalize_to_first_person(item["answer"])
        for item in context_data
    ]
    notes_bullet_block = "\n".join(f"- {note}" for note in knowledge_notes)

    # Stage 4A: FLAN-T5 Factual Rule Distillation
    flan_prompt = (
        "Instruction: Summarize the key investment rules and formulas from Peter Lynch's notes below "
        f"to answer: '{query}'\n\n"
        f"Notes:\n{notes_bullet_block}\n\n"
        "Key Investment Rules:"
    )
    if hasattr(llm, "distill_with_flan_t5"):
        flan_distilled = _normalize_to_first_person(llm.distill_with_flan_t5(flan_prompt))
    else:
        flan_distilled = _normalize_to_first_person(
            llm(flan_prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=temperature)[0]["generated_text"]
        )

    portfolio_section = _build_portfolio_grounding(query, portfolio_df)

    # Stage 4B: Conversational AI Agent Synthesis
    if hasattr(llm, "synthesize_conversational_response") and getattr(llm, "use_chat_agent", False):
        synthesized_body = llm.synthesize_conversational_response(
            question=query,
            flan_distilled_facts=flan_distilled,
            knowledge_notes=knowledge_notes,
            portfolio_context="",
            temperature=temperature,
        )
    else:
        synthesized_body = flan_distilled

    response_sections = [synthesized_body]
    if portfolio_section:
        response_sections.append(portfolio_section)

    final_answer = "\n\n".join(response_sections).strip()
    telemetry["flan_t5_distilled_facts"] = flan_distilled
    telemetry["synthesized_response"] = synthesized_body

    if return_telemetry:
        return final_answer, context_data, telemetry
    return final_answer, context_data
