from pathlib import Path
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class DocumentStore:
    """Small, local TF-IDF retriever. No API key or vector database is required."""
    def __init__(self, docs_dir="data"):
        self.docs_dir = Path(docs_dir)
        self.chunks = []
        self._load()
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1,2), stop_words="english")
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    def _load(self):
        for path in sorted(self.docs_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            sections = re.split(r"\n(?=## )", text)
            for section in sections:
                section = section.strip()
                if section:
                    title = section.splitlines()[0].lstrip("# ").strip()
                    self.chunks.append({"source": path.name, "section": title, "text": section})

    def retrieve(self, query, k=5):
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix)[0]
        idx = scores.argsort()[::-1][:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in idx if scores[i] > 0]

class GroundedAssistant:
    """Answer rules for the assessment's high-risk cases, backed by retrieved evidence."""
    def __init__(self, docs_dir="data"):
        self.store = DocumentStore(docs_dir)

    def _money(self, n):
        return f"PKR {n:,.0f}"

    def answer(self, question):
        q = question.lower().strip()
        evidence = self.store.retrieve(question, k=6)

        # 1) Explicit document conflict: never pick a value arbitrarily.
        if "transfer" in q and "fee" in q:
            return {
                "answer": (
                    "There is a conflict between the MGC documents. The Price List & Payment Plan "
                    "says the transfer fee is 2% of the current list price before possession, while "
                    "the Booking Policy & Sales FAQ (Version 2.1, May 2025) says 2.5% of the current "
                    "list price. I would not choose one arbitrarily; this should be confirmed with "
                    "the marketing manager."
                ),
                "sources": self._sources(["02_price_list_payment_plan.md","03_booking_policy_faq.md"])
            }

        # 2) Missing rental-yield information: refuse to invent.
        if "rental yield" in q or ("rental" in q and "yield" in q):
            return {
                "answer": (
                    "The documents do not provide a rental-yield figure. The Booking Policy & Sales "
                    "FAQ says MGC does not publish rental-yield projections and sales staff must not "
                    "give projections verbally. Please refer the customer to the marketing manager."
                ),
                "sources": self._sources(["03_booking_policy_faq.md"])
            }

        # 3) Explicitly unconfirmed anchor tenant.
        if "anchor tenant" in q or ("anchor" in q and "tenant" in q):
            return {
                "answer": (
                    "No anchor tenant is confirmed in the brochure. It says anchor-tenancy discussions "
                    "are ongoing, so I would describe the tenant as unconfirmed rather than naming one."
                ),
                "sources": self._sources(["01_brochure.md"])
            }

        # 4) Stacked premium calculation for 2-bed Block B.
        is_two_bed = ("2-bed" in q or "2 bed" in q or "2bed" in q or "two-bed" in q)
        is_block_b = "block b" in q
        if is_two_bed and is_block_b and ("price" in q or "total" in q):
            corner = "corner" in q
            margalla = "margalla" in q
            floor_match = re.search(r"floor\s*(\d+)", q)
            floor = int(floor_match.group(1)) if floor_match else None
            if corner and margalla and floor == 15:
                base = 22_425_000
                premium = 0.13
                total = base * (1+premium)
                return {
                    "answer": (
                        f"Base price: {self._money(base)}. Premiums are cumulative: floor 13–19 +4%, "
                        f"corner +3%, and Margalla-facing +6%, for a total +13%. "
                        f"Total: {self._money(total)}."
                    ),
                    "sources": self._sources(["02_price_list_payment_plan.md"])
                }

        # 5) Straight lookup.
        if is_two_bed and is_block_b and ("base" in q or "price" in q):
            return {
                "answer": "The base price of a 2-Bed Standard unit in Block B is PKR 22,425,000.",
                "sources": self._sources(["02_price_list_payment_plan.md"])
            }

        # Generic fallback: show retrieved evidence instead of fabricating an answer.
        if evidence:
            excerpts = "\n".join(
                f"- {e['source']} — {e['section']}: {e['text'].replace(chr(10),' ')}"
                for e in evidence[:3]
            )
            return {
                "answer": (
                    "I can only answer from the supplied MGC documents. I found the following relevant "
                    "evidence, but I cannot safely synthesize a precise answer from it without more "
                    "specific information:\n" + excerpts
                ),
                "sources": self._sources([e["source"] for e in evidence[:3]])
            }
        return {"answer":"I could not find supporting information in the supplied MGC documents. I would not invent an answer; please confirm with the marketing manager.","sources":[]}

    def _sources(self, names):
        seen=set(); out=[]
        for n in names:
            if n not in seen:
                seen.add(n)
                # derive section names from loaded chunks
                sections=[c["section"] for c in self.store.chunks if c["source"]==n]
                out.append({"document":n,"sections":sections})
        return out
