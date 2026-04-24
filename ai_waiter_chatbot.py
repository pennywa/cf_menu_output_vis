import re
from collections import Counter
from dataclasses import dataclass
from math import log
from pathlib import Path


MENU_FILE = Path("cheesecake_factory_menu.txt")


@dataclass
class MenuItem:
    section: str
    name: str
    price: str
    source_line: str


def normalize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def load_menu_items(path: Path) -> list[MenuItem]:
    if not path.exists():
        raise FileNotFoundError(f"Menu file not found: {path}")

    items: list[MenuItem] = []
    current_section = "MENU"

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if "— $" in line:
            name, price = line.split("—", 1)
            items.append(
                MenuItem(
                    section=current_section,
                    name=name.strip(),
                    price=price.strip(),
                    source_line=line,
                )
            )
            continue

        is_section = line.isupper() and "— $" not in line and len(line) > 2
        if is_section:
            current_section = line

    return items


class TinyRetriever:
    def __init__(self, items: list[MenuItem]) -> None:
        self.items = items
        self.doc_tokens = [normalize(f"{i.section} {i.name}") for i in items]
        self.df = Counter()
        for toks in self.doc_tokens:
            for t in set(toks):
                self.df[t] += 1

    def _score(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        if not doc_tokens:
            return 0.0
        tf = Counter(doc_tokens)
        n_docs = len(self.doc_tokens)
        score = 0.0
        for q in query_tokens:
            if q not in tf:
                continue
            idf = log((n_docs + 1) / (self.df[q] + 1)) + 1.0
            score += tf[q] * idf
        return score

    def retrieve(self, query: str, top_k: int = 6) -> list[MenuItem]:
        q_tokens = normalize(query)
        ranked = sorted(
            zip(self.items, self.doc_tokens),
            key=lambda pair: self._score(q_tokens, pair[1]),
            reverse=True,
        )
        best = [item for item, _ in ranked[:top_k]]
        return [i for i in best if i]


def answer_without_rag(user_query: str) -> str:
    q = user_query.lower()
    if any(k in q for k in ["cheesecake", "dessert", "sweet"]):
        return (
            "Without RAG: I do not have direct access to your uploaded menu, "
            "but Cheesecake Factory usually has many cheesecake flavors, "
            "classic desserts, and seasonal sweets."
        )
    if any(k in q for k in ["vegan", "vegetarian", "salad", "healthy", "skinny"]):
        return (
            "Without RAG: I cannot see the exact menu text here, but the restaurant "
            "typically offers salads, veggie options, and lighter items."
        )
    if any(k in q for k in ["price", "cost", "how much", "$"]):
        return (
            "Without RAG: I cannot provide exact prices because I am not retrieving "
            "from your actual menu data in this mode."
        )

    return (
        "Without RAG: I can provide a general restaurant-style answer, but this mode "
        "is not grounded in your Cheesecake Factory menu file."
    )


def answer_with_rag(user_query: str, retriever: TinyRetriever) -> str:
    hits = retriever.retrieve(user_query, top_k=6)
    if not hits:
        return (
            "With RAG: I could not find a direct match in the menu text. "
            "Try asking about a category like pasta, salads, burgers, or cheesecakes."
        )

    lines = []
    seen = set()
    for h in hits:
        key = (h.section, h.name, h.price)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {h.name} ({h.section}) {h.price}")
        if len(lines) == 5:
            break

    if not lines:
        return (
            "With RAG: I found relevant data but could not format unique results."
        )

    return "With RAG: Based on your menu file, here are relevant items:\n" + "\n".join(lines)


def main() -> None:
    items = load_menu_items(MENU_FILE)
    retriever = TinyRetriever(items)

    print("AI Waiter Chatbot (Cheesecake Factory)")
    print("Type a question about the menu. Type 'exit' to quit.\n")

    while True:
        user_query = input("You: ").strip()
        if not user_query:
            continue
        if user_query.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        no_rag = answer_without_rag(user_query)
        rag = answer_with_rag(user_query, retriever)

        print("\n--- Response 1: Without RAG ---")
        print(no_rag)
        print("\n--- Response 2: With RAG ---")
        print(rag)
        print()


if __name__ == "__main__":
    main()
