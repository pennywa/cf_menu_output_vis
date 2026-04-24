import gradio as gr

from ai_waiter_chatbot import (
    MENU_FILE,
    TinyRetriever,
    answer_without_rag,
    load_menu_items,
)
from embedding_vis import (
    build_embedding_figure,
    build_embedding_figure_3d,
    build_pyvis_network_html,
)


items = load_menu_items(MENU_FILE)
retriever = TinyRetriever(items)


def _answer_with_rag_top_k(user_query: str, top_k: int) -> str:
    hits = retriever.retrieve(user_query, top_k=top_k)
    if not hits:
        return (
            "With RAG: I could not find a direct match in the menu text. "
            "Try asking about a category like pasta, salads, burgers, or cheesecakes."
        )

    lines = []
    seen: set[tuple[str, str, str]] = set()
    for h in hits:
        key = (h.section, h.name, h.price)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {h.name} ({h.section}) {h.price}")
        if len(lines) == top_k:
            break

    return "With RAG: Based on your menu file, here are relevant items:\n" + "\n".join(lines)


def ask_waiter(user_query: str, top_k: int):
    query = (user_query or "").strip()
    if not query:
        return (
            "Please enter a menu question.",
            "Please enter a menu question.",
            build_embedding_figure("menu", items, retriever, int(top_k)),
            build_embedding_figure_3d("menu", items, retriever, int(top_k)),
            build_pyvis_network_html("menu", items, retriever, int(top_k)),
        )

    no_rag = answer_without_rag(query)
    with_rag = _answer_with_rag_top_k(query, int(top_k))
    fig_2d = build_embedding_figure(query, items, retriever, int(top_k))
    fig_3d = build_embedding_figure_3d(query, items, retriever, int(top_k))
    pyvis_html = build_pyvis_network_html(query, items, retriever, int(top_k))
    return no_rag, with_rag, fig_2d, fig_3d, pyvis_html


demo = gr.Interface(
    fn=ask_waiter,
    inputs=[
        gr.Textbox(
            lines=2,
            label="Ask the AI Waiter",
            placeholder="Example: What pasta dishes do you have under $26?",
        ),
        gr.Slider(
            minimum=1,
            maximum=12,
            value=6,
            step=1,
            label="Top-k retrievals",
        ),
    ],
    outputs=[
        gr.Textbox(label="Response 1: Without RAG"),
        gr.Textbox(label="Response 2: With RAG"),
        gr.Plot(label="Embedding Nodes (Interactive)"),
        gr.Plot(label="3D Embedding Space (Interactive)"),
        gr.HTML(label="PyVis Network Explorer"),
    ],
    title="Cheesecake Factory AI Waiter (RAG vs No RAG)",
    description=(
        "Ask a question about the menu and compare responses. "
        "The first response is generic (no retrieval). "
        "The second response is retrieval-grounded from cheesecake_factory_menu.txt. "
        "The 2D/3D plots color nodes by category and highlight retrieved nodes. "
        "Use the PyVis graph to travel through node connections."
    ),
)


if __name__ == "__main__":
    demo.launch()
