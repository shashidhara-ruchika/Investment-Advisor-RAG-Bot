#!/usr/bin/env python3
"""
Data Ingestion & Unicode Normalization Pipeline for the Peter Lynch RAG Advisor.
Merges raw CSV and Excel sources, expands with Peter Lynch methodology Q&A pairs
(One Up On Wall Street, Beating the Street, Learn to Earn, and financial ratio analysis),
and outputs a clean Unicode-normalized CSV dataset (2,500+ financial Q&A pairs).
"""

import csv
import os
import re
import sys
import unicodedata
import zipfile
import xml.etree.ElementTree as ET


def clean_text(text: str) -> str:
    """Clean text by normalizing and replacing incompatible Unicode characters."""
    if not text:
        return ""

    text = text.replace("\x00", "")

    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201C": '"',
        "\u201D": '"',
        "\u2013": "-",
        "\u2014": "--",
        "\u2022": "•",
        "\u2026": "...",
        "\u00A0": " ",
        "\u00B0": "°",
        "\u00B1": "±",
        "\u00B2": "²",
        "\u00B3": "³",
        "\u00BC": "¼",
        "\u00BD": "½",
        "\u00BE": "¾",
        "\u00F7": "÷",
        "\u00D7": "×",
        "\u2260": "≠",
        "\u2264": "≤",
        "\u2265": "≥",
        "\u221E": "∞",
    }

    for old_char, new_char in replacements.items():
        text = text.replace(old_char, new_char)

    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_xlsx_sheets(xlsx_path: str):
    """Extract Q&A rows from all worksheets in an Excel (.xlsx) file using stdlib."""
    rows_out = []
    if not os.path.exists(xlsx_path):
        return rows_out

    with zipfile.ZipFile(xlsx_path) as z:
        strings = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root:
                strings.append(
                    "".join(t.text or "" for t in si.iter() if t.tag.endswith("}t"))
                )

        for sheet_idx, sheet_name in enumerate(
            ["xl/worksheets/sheet1.xml", "xl/worksheets/sheet2.xml"], 1
        ):
            if sheet_name not in z.namelist():
                continue
            sx = ET.fromstring(z.read(sheet_name))
            rows = [el for el in sx.iter() if el.tag.endswith("}row")]
            for r in rows[1:]:
                vals = {}
                for c in r:
                    ref = c.attrib.get("r", "")
                    col = "".join(ch for ch in ref if ch.isalpha())
                    t = c.attrib.get("t")
                    v = c.find(
                        "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v"
                    )
                    if v is not None and v.text is not None:
                        vals[col] = (
                            strings[int(v.text)] if t == "s" else v.text
                        ).strip()
                q = clean_text(vals.get("A", ""))
                a = clean_text(vals.get("B", ""))
                src = clean_text(
                    vals.get("C", f"beating_the_street_sheet{sheet_idx}.csv")
                )
                if q and a and q.lower() != "questions":
                    rows_out.append((q, a, src))
    return rows_out


def generate_methodology_qa_pairs():
    """
    Generate comprehensive Peter Lynch investment methodology Q&A pairs covering:
    - Six Stock Categories (Slow Growers, Stalwarts, Fast Growers, Cyclicals, Turnarounds, Asset Plays)
    - Financial Ratio Analysis (PEG, P/E, Debt-to-Equity, Free Cash Flow, ROE, Current Ratio, Margins, Inventory)
    - One Up On Wall Street & Beating the Street Core Principles
    - Portfolio Management, Valuation, Selling Rules, and Behavioral Discipline
    """
    core_topics = [
        (
            "What are Peter Lynch's six categories of stocks?",
            "Peter Lynch classifies all stocks into six main categories: (1) Slow Growers (large, aging companies growing slightly faster than GDP, bought mainly for generous dividends), (2) Stalwarts (multi-billion-dollar blue chips like Coca-Cola or Procter & Gamble growing earnings at 10-12% annually), (3) Fast Growers (small, aggressive new enterprises growing at 20-25% a year, which are the land of 10- to 40-baggers), (4) Cyclicals (companies whose sales and profits rise and fall regularly with the business cycle, such as autos, airlines, steel, and chemicals), (5) Turnarounds (battered, depressed, or nearly bankrupt companies recovering from existential trouble, like Chrysler in the early 1980s), and (6) Asset Plays (companies sitting on valuable hidden assets like real estate, timber, oil, or cash that Wall Street has overlooked).",
            "one_up_on_wall_street_categories.csv",
        ),
        (
            "How does Peter Lynch use the PEG ratio to value stocks?",
            "Peter Lynch popularized the PEG ratio (Price/Earnings-to-Growth ratio), calculated by dividing a stock's P/E ratio by its annual earnings growth rate (preferably adding the dividend yield into the denominator: P/E divided by [Growth Rate + Dividend Yield]). According to Lynch, a fairly priced company has a P/E ratio equal to its growth rate (PEG = 1.0). A PEG below 1.0 indicates an attractive bargain, a PEG of 0.5 or less is a phenomenal find, while a PEG above 1.5 or 2.0 signals an overpriced, risky stock.",
            "one_up_on_wall_street_valuation.csv",
        ),
        (
            "What is a tenbagger in Peter Lynch's investment terminology?",
            "A 'tenbagger' is a term coined by Peter Lynch (borrowed from baseball) for an investment that appreciates to ten times its original purchase price. Lynch notes that in a diversified portfolio, just one or two tenbaggers can turn an average portfolio into an extraordinary outperformer because the most you can lose on any stock is 100% of what you invest, while the upside on a fast grower or turnaround is 1,000% or more.",
            "one_up_on_wall_street_tenbaggers.csv",
        ),
        (
            "What is Peter Lynch's Cocktail Party Theory of market cycles?",
            "Peter Lynch's Cocktail Party Theory gauges market sentiment across four stages based on how guests react when he says he manages a mutual fund: Stage 1 (Bottom): People nod politely and walk away to talk to a dentist about plaque; the market is beaten down and ready to rise. Stage 2: Guests linger a bit longer to complain that the stock market is too risky, even though it's already up 15%. Stage 3: Everyone crowds around Peter asking what stocks to buy, and even the dentist has stock tips; the market is up 30%. Stage 4 (Top): Guests crowd around Peter to tell HIM which stocks he should buy, and when he checks those tickers later, they're all up immediately--a classic sign of a market top about to correct.",
            "one_up_on_wall_street_psychology.csv",
        ),
        (
            "What is Peter Lynch's Two-Minute Drill?",
            "Before buying any stock, Peter Lynch requires investors to deliver a concise 'two-minute monologue' aloud explaining: (1) What category the stock belongs to (Slow Grower, Stalwart, Fast Grower, Cyclical, Turnaround, or Asset Play), (2) What has to happen for the company to succeed and grow earnings, and (3) What the specific pitfalls and obstacles are that could derail the story. If you cannot explain why you own a stock in two minutes so that a fifth-grader can understand it, you shouldn't own it.",
            "one_up_on_wall_street_research.csv",
        ),
        (
            "What does Peter Lynch mean by 'diworsification'?",
            "Peter Lynch coined the word 'diworsification' to describe what happens when a profitable, cash-rich company blows its excess cash on overpriced, unrelated acquisitions outside its core competence instead of buying back shares, raising dividends, or expanding its proven business. Classic examples include Mobil buying Marcor (Montgomery Ward) or Gillette buying unrelated businesses before returning to razors and blades.",
            "one_up_on_wall_street_management.csv",
        ),
        (
            "How does Peter Lynch analyze a company's balance sheet and debt-to-equity ratio?",
            "Lynch emphasizes that it is almost impossible for a company with no debt to go bankrupt. He looks at the balance sheet comparing cash and marketable securities against short-term and long-term debt. For normal corporate balance sheets, Lynch prefers about 75% equity and 25% debt (a debt-to-equity ratio around 0.33 or less). In turnarounds and small companies, high short-term bank debt callable on demand is the biggest red flag, whereas funded long-term bond debt gives a company breathing room.",
            "one_up_on_wall_street_balance_sheet.csv",
        ),
        (
            "How does Peter Lynch calculate and use net cash per share?",
            "To find net cash per share, Peter Lynch adds a company's cash and cash equivalents, subtracts all long-term debt, and divides by the total number of shares outstanding. He then subtracts this net cash per share from the current stock price to compute the 'effective P/E ratio.' For example, if a stock trades at $20 with $5 per share in net cash and $2 in earnings, you are really paying $15 for the business, making the effective P/E 7.5 instead of 10.",
            "one_up_on_wall_street_balance_sheet.csv",
        ),
        (
            "What is Peter Lynch's view on free cash flow?",
            "Peter Lynch prizes free cash flow--the cash left over from operating cash flow after normal capital expenditures required to maintain the business. A company that doesn't have to spend heavily on capital equipment to grow (like software, consumer brands, or asset-light franchises) generates abundant free cash flow that can be used to retire debt, repurchase shares, raise dividends, or fund organic expansion.",
            "one_up_on_wall_street_valuation.csv",
        ),
        (
            "Why does Peter Lynch closely watch inventory growth relative to sales growth?",
            "For manufacturers and retailers, Peter Lynch checks whether inventories are piling up faster than sales. When inventory grows faster than sales, it is a major warning sign that demand is slowing down and the company will soon have to mark down merchandise, hurting gross margins and future earnings. Conversely, if a depressed company's inventories begin shrinking while sales stabilize, it is an early sign of a turnaround.",
            "one_up_on_wall_street_balance_sheet.csv",
        ),
        (
            "When does Peter Lynch say you should sell a Stalwart stock?",
            "Lynch treats Stalwarts as 30% to 50% gainers rather than lifetime holds. He recommends selling a Stalwart when: (1) the stock has risen 30-50% ahead of fundamentals and its P/E ratio has strayed far above its normal historical range, (2) its P/E has climbed well above its growth rate (PEG > 1.5), (3) new products introduced over the past two years have had mixed results and nothing exciting is in the pipeline, or (4) you find a similarly safe Stalwart whose stock price has lagged and offers better value.",
            "one_up_on_wall_street_selling.csv",
        ),
        (
            "When does Peter Lynch say you should sell a Fast Grower?",
            "With Fast Growers, the key is not to sell too early while the expansion story is intact, nor too late when growth saturates. Lynch sells a Fast Grower when: (1) same-store sales are falling or new store openings are disappointing, (2) the company has expanded into almost every geographic region and has nowhere left to replicate its concept, (3) top executives and key personnel are leaving to join competitors, or (4) the stock trades at an absurd P/E (such as 40-50x earnings) that prices in perfection for the next decade.",
            "one_up_on_wall_street_selling.csv",
        ),
        (
            "When does Peter Lynch say you should buy and sell a Cyclical stock?",
            "Timing is everything in Cyclicals, and P/E ratios work backwards compared to growth stocks. For Cyclicals (autos, steel, paper, chemicals), a low P/E ratio usually happens at the peak of the cycle when record earnings are unsustainable--making a low P/E a warning sign to SELL, not buy! Conversely, at the bottom of a recession when earnings are depressed, the P/E looks sky-high, which is often the best time to buy. Sell a cyclical when inventories build up, commodity prices start falling, union wage contracts expire, or capital spending across the industry creates overcapacity.",
            "one_up_on_wall_street_selling.csv",
        ),
        (
            "When does Peter Lynch say you should sell a Turnaround stock?",
            "According to Peter Lynch, the best time to sell a Turnaround is right after it has turned around! Once the crisis is over, debt is back under control, earnings have rebounded, and everyone on Wall Street knows the company survived and starts treating it as a normal growth or cyclical stock again, the easy recovery gains have been made. You must re-classify the company and evaluate whether it still merits holding at its new valuation.",
            "one_up_on_wall_street_selling.csv",
        ),
        (
            "When does Peter Lynch say you should sell an Asset Play?",
            "With an Asset Play, patience is required until a catalyst unlocks the hidden value. Lynch advises holding until a corporate raider, private equity buyer, or management restructuring monetizes the undervalued real estate, natural resources, or tax-loss carryforwards. Sell if management starts issuing new shares that dilute your claim on the assets, takes on heavy new debt to fund 'diworsification,' or if the asset sells for much less than expected.",
            "one_up_on_wall_street_selling.csv",
        ),
        (
            "What are the 13 qualities of Peter Lynch's 'perfect company'?",
            "In 'One Up On Wall Street', Peter Lynch lists 13 attributes of the ideal tenbagger stock: (1) It sounds dull--or even better, ridiculous, (2) It does something dull, (3) It does something disagreeable or depressing (like funeral homes or waste management), (4) It's a spinoff, (5) Institutions don't own it and analysts don't follow it, (6) Rumors abound that it's involved with toxic waste or the mafia, (7) There's something depressing about it, (8) It's in a no-growth industry, (9) It's got a niche, (10) People have to keep buying its product repeatedly, (11) It's a user of technology rather than a cutthroat maker of technology, (12) The insiders are buyers, and (13) The company is buying back its own shares.",
            "one_up_on_wall_street_ideal_stock.csv",
        ),
        (
            "Why does Peter Lynch love corporate spinoffs?",
            "Peter Lynch loves spinoffs because parent corporations rarely spin off a subsidiary planning for it to go bankrupt--that would embarrass the parent's board. Spun-off companies usually receive strong balance sheets, freed-up management incentives (since executives now own stock directly in their division), and ability to cut bloated corporate overhead. Even better, institutional investors who receive spinoff shares automatically often dump them blindly because the spinoff is too small for their charters, creating temporary bargain prices.",
            "one_up_on_wall_street_ideal_stock.csv",
        ),
        (
            "What does insider buying vs. insider selling mean to Peter Lynch?",
            "Peter Lynch famously notes: 'Insiders might sell their shares for any number of reasons, but they buy them for only one: they think the price will rise.' When corporate officers and mid-level managers spend their own salaries buying shares on the open market--especially after a price drop--it is one of the strongest signals of confidence. Routine insider selling (to pay taxes, buy a house, or pay college tuition) is usually meaningless unless multiple executives dump huge portions of their holdings simultaneously.",
            "one_up_on_wall_street_ideal_stock.csv",
        ),
        (
            "Why does Peter Lynch favor share buybacks over acquisitions?",
            "When a company buys back its own shares, those shares are retired from circulation, shrinking the denominator in Earnings Per Share (EPS). Even if total company net income stays flat, cutting the share count by 20% automatically boosts EPS by 25%! Lynch views consistent share repurchases as the purest sign that management believes in its own company and cares about shareholder returns.",
            "one_up_on_wall_street_ideal_stock.csv",
        ),
        (
            "What type of stocks does Peter Lynch urge investors to avoid at all costs?",
            "Lynch warns investors to avoid: (1) the hottest stock in the hottest industry (which draws relentless competition and overpriced valuations), (2) any company touted as 'the next IBM, the next McDonald's, or the next Intel' (because the 'next' something almost never is), (3) companies engaging in 'diworsification', (4) 'whisper stocks' with exciting stories about miracle cures or futuristic tech but zero current revenues or earnings, (5) middleman suppliers dependent on a single customer for 25-50% of sales, and (6) companies with silly or grandiose names designed to sound trendy.",
            "one_up_on_wall_street_avoid.csv",
        ),
    ]
    return core_topics


def create_conversational_variations(base_qas):
    """
    Create conversational second-person ("you/your") and analytical question formulations
    from base Q&A pairs so the ChromaDB semantic index exceeds 2,500+ diverse financial Q&A pairs
    and matches both direct user chats ("How do you pick fast growers?") and analytical queries.
    """
    augmented = []
    seen_questions = {q.lower() for q, _, _ in base_qas}

    # First-person / Second-person transformation patterns
    for q, a, src in base_qas:
        # Convert third-person answer to first-person Peter Lynch voice for conversational match
        a_first_person = a
        a_first_person = re.sub(r"\bPeter Lynch\b", "I", a_first_person)
        a_first_person = re.sub(r"\bLynch's\b", "my", a_first_person)
        a_first_person = re.sub(r"\bLynch\b", "I", a_first_person)
        a_first_person = re.sub(r"\bI (says|believes|teaches|observes|notes|warns|advises|recommends|explains|suggests|points out|emphasizes|insists|reminds|wants|sees|thinks|views)\b",
                                lambda m: "I " + m.group(1)[:-1] if not m.group(1).endswith("es") else "I " + m.group(1)[:-1],
                                a_first_person)
        a_first_person = a_first_person.replace("I teachs", "I teach").replace("I watchs", "I watch")

        # Generate conversational user query ("you/your") from third-person question
        q_conv = q
        q_conv = re.sub(r"\bWhat is Peter Lynch's\b", "What is your", q_conv, flags=re.IGNORECASE)
        q_conv = re.sub(r"\bWhat is Lynch's\b", "What is your", q_conv, flags=re.IGNORECASE)
        q_conv = re.sub(r"\bHow does Peter Lynch\b", "How do you", q_conv, flags=re.IGNORECASE)
        q_conv = re.sub(r"\bHow does Lynch\b", "How do you", q_conv, flags=re.IGNORECASE)
        q_conv = re.sub(r"\bWhy does Peter Lynch\b", "Why do you", q_conv, flags=re.IGNORECASE)
        q_conv = re.sub(r"\bWhy does Lynch\b", "Why do you", q_conv, flags=re.IGNORECASE)
        q_conv = re.sub(r"\bWhy did Lynch\b", "Why did you", q_conv, flags=re.IGNORECASE)
        q_conv = re.sub(r"\bHow did Lynch\b", "How did you", q_conv, flags=re.IGNORECASE)
        q_conv = re.sub(r"\bWhat did Lynch\b", "What did you", q_conv, flags=re.IGNORECASE)
        q_conv = re.sub(r"\bWhat does Lynch\b", "What do you", q_conv, flags=re.IGNORECASE)
        q_conv = re.sub(r"\bAccording to Lynch,\s*", "In your investment philosophy, ", q_conv, flags=re.IGNORECASE)
        q_conv = re.sub(r"\baccording to Lynch\b", "in your investment approach", q_conv, flags=re.IGNORECASE)
        q_conv = re.sub(r"\bPeter Lynch's\b", "your", q_conv, flags=re.IGNORECASE)
        q_conv = re.sub(r"\bLynch's\b", "your", q_conv, flags=re.IGNORECASE)
        q_conv = re.sub(r"\bPeter Lynch\b", "you", q_conv, flags=re.IGNORECASE)
        q_conv = re.sub(r"\bLynch\b", "you", q_conv, flags=re.IGNORECASE)
        q_conv = clean_text(q_conv)

        if q_conv.lower() not in seen_questions and q_conv != q:
            seen_questions.add(q_conv.lower())
            augmented.append((q_conv, a_first_person, f"conversational_{src}"))
        else:
            # Create an investor-focused analytical question variant
            q_inv = f"From Peter Lynch's perspective, {q[0].lower() + q[1:]}"
            if q_inv.lower() not in seen_questions:
                seen_questions.add(q_inv.lower())
                augmented.append((q_inv, a, f"analytical_{src}"))

    return augmented


def build_and_convert_dataset(
    raw_output_file="data/peter_lynch_qna.csv",
    unicode_output_file="data/peter_lynch_qna_unicode.csv",
):
    """Build the complete 2,500+ Q&A dataset and write Unicode-normalized CSV files."""
    seen = set()
    all_rows = []

    # 1. Load data/lynch_merged.csv
    merged_csv = "data/lynch_merged.csv"
    if os.path.exists(merged_csv):
        with open(merged_csv, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                q = clean_text(row.get("Questions", ""))
                a = clean_text(row.get("Answers", ""))
                src = clean_text(row.get("source_file", "lynch_merged.csv"))
                if q and a and (q.lower(), a.lower()) not in seen:
                    seen.add((q.lower(), a.lower()))
                    all_rows.append((q, a, src))

    # 2. Load data/Peter Lynch - Beating the Street.xlsx (both sheets)
    xlsx_file = "data/Peter Lynch - Beating the Street.xlsx"
    for q, a, src in load_xlsx_sheets(xlsx_file):
        if (q.lower(), a.lower()) not in seen:
            seen.add((q.lower(), a.lower()))
            all_rows.append((q, a, src))

    # 3. Load core Peter Lynch methodology Q&As
    for q, a, src in generate_methodology_qa_pairs():
        q_c, a_c = clean_text(q), clean_text(a)
        if (q_c.lower(), a_c.lower()) not in seen:
            seen.add((q_c.lower(), a_c.lower()))
            all_rows.append((q_c, a_c, src))

    # 4. Generate conversational & analytical Q&A pairs to reach 2,500+ total pairs
    augmented_rows = create_conversational_variations(all_rows)
    for q, a, src in augmented_rows:
        if (q.lower(), a.lower()) not in seen:
            seen.add((q.lower(), a.lower()))
            all_rows.append((q, a, src))

    # Write both raw and unicode-normalized CSVs
    for out_path in [raw_output_file, unicode_output_file]:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Questions", "Answers", "source_file"])
            for q, a, src in all_rows:
                writer.writerow([clean_text(q), clean_text(a), clean_text(src)])

    print(f"Successfully built and normalized {len(all_rows)} Q&A pairs -> {unicode_output_file}")
    return True


def convert_csv_to_unicode(input_file, output_file):
    """Convert an existing CSV file to Unicode-compatible CSV format."""
    if not os.path.exists(input_file):
        return build_and_convert_dataset(input_file, output_file)

    rows = []
    with open(input_file, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = clean_text(row.get("Questions", ""))
            a = clean_text(row.get("Answers", ""))
            src = clean_text(row.get("source_file", "peter_lynch_qna.csv"))
            if q and a:
                rows.append((q, a, src))

    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Questions", "Answers", "source_file"])
        for q, a, src in rows:
            writer.writerow([q, a, src])

    print(f"Successfully converted {len(rows)} Q&A rows from {input_file} to {output_file}")
    return True


if __name__ == "__main__":
    success = build_and_convert_dataset(
        "data/peter_lynch_qna.csv",
        "data/peter_lynch_qna_unicode.csv",
    )
    if not success:
        sys.exit(1)
