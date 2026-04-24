from collections import Counter
from math import log

import numpy as np
import plotly.graph_objects as go
from pyvis.network import Network

from ai_waiter_chatbot import MenuItem, TinyRetriever, normalize


PALETTE = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#EECA3B",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
]


def _build_tfidf_vectors(
    items: list[MenuItem], retriever: TinyRetriever
) -> tuple[list[str], np.ndarray]:
    vocab = sorted(retriever.df.keys())
    token_to_idx = {t: i for i, t in enumerate(vocab)}
    n_docs = len(items)
    matrix = np.zeros((n_docs, len(vocab)), dtype=float)

    for row_idx, doc_tokens in enumerate(retriever.doc_tokens):
        tf = Counter(doc_tokens)
        for token, count in tf.items():
            col_idx = token_to_idx.get(token)
            if col_idx is None:
                continue
            idf = log((n_docs + 1) / (retriever.df[token] + 1)) + 1.0
            matrix[row_idx, col_idx] = count * idf

    return vocab, matrix


def _query_vector(query: str, vocab: list[str], retriever: TinyRetriever) -> np.ndarray:
    token_to_idx = {t: i for i, t in enumerate(vocab)}
    vec = np.zeros((len(vocab),), dtype=float)
    q_tokens = Counter(normalize(query))
    n_docs = max(len(retriever.doc_tokens), 1)

    for token, count in q_tokens.items():
        if token not in token_to_idx:
            continue
        idf = log((n_docs + 1) / (retriever.df[token] + 1)) + 1.0
        vec[token_to_idx[token]] = count * idf

    return vec


def _project_2d(matrix: np.ndarray, query_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    full = np.vstack([matrix, query_vec.reshape(1, -1)])
    full = full - full.mean(axis=0, keepdims=True)

    # SVD gives a stable 2D projection without extra dependencies.
    _, _, vt = np.linalg.svd(full, full_matrices=False)
    axes = vt[:2].T
    coords = full @ axes

    return coords[:-1], coords[-1]


def _project_nd(
    matrix: np.ndarray, query_vec: np.ndarray, dims: int
) -> tuple[np.ndarray, np.ndarray]:
    full = np.vstack([matrix, query_vec.reshape(1, -1)])
    full = full - full.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(full, full_matrices=False)
    take = min(dims, vt.shape[0])
    axes = vt[:take].T
    coords = full @ axes
    if take < dims:
        pad = np.zeros((coords.shape[0], dims - take))
        coords = np.hstack([coords, pad])
    return coords[:-1], coords[-1]


def _section_color_map(items: list[MenuItem]) -> dict[str, str]:
    sections = sorted({item.section for item in items})
    return {section: PALETTE[i % len(PALETTE)] for i, section in enumerate(sections)}


def build_embedding_figure(
    query: str, items: list[MenuItem], retriever: TinyRetriever, top_k: int = 6
) -> go.Figure:
    vocab, matrix = _build_tfidf_vectors(items, retriever)
    q_vec = _query_vector(query, vocab, retriever)
    item_xy, query_xy = _project_2d(matrix, q_vec)

    retrieved = retriever.retrieve(query, top_k=top_k)
    retrieved_ids = {
        (r.section, r.name, r.price)
        for r in retrieved
    }

    section_colors = _section_color_map(items)
    by_section: dict[str, dict[str, list]] = {}
    hit_x, hit_y, hit_text = [], [], []

    for idx, item in enumerate(items):
        payload = f"{item.name}<br>{item.section}<br>{item.price}"
        target = (item.section, item.name, item.price)
        point_x = float(item_xy[idx, 0])
        point_y = float(item_xy[idx, 1])

        if item.section not in by_section:
            by_section[item.section] = {"x": [], "y": [], "text": []}
        by_section[item.section]["x"].append(point_x)
        by_section[item.section]["y"].append(point_y)
        by_section[item.section]["text"].append(payload)

        if target in retrieved_ids:
            hit_x.append(point_x)
            hit_y.append(point_y)
            hit_text.append(payload)

    fig = go.Figure()
    for section, payload in by_section.items():
        fig.add_trace(
            go.Scatter(
                x=payload["x"],
                y=payload["y"],
                mode="markers",
                name=section,
                marker={
                    "size": 8,
                    "opacity": 0.50,
                    "color": section_colors[section],
                },
                text=payload["text"],
                hovertemplate="%{text}<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=hit_x,
            y=hit_y,
            mode="markers",
            name="Retrieved nodes",
            marker={
                "size": 13,
                "opacity": 1.0,
                "color": "#111111",
                "symbol": "diamond-open",
                "line": {"width": 2},
            },
            text=hit_text,
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[float(query_xy[0])],
            y=[float(query_xy[1])],
            mode="markers",
            name="Your question",
            marker={"size": 16, "symbol": "star", "color": "#54A24B"},
            text=[query],
            hovertemplate="Query: %{text}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Interactive Embedding Space (Top-k retrieved: {top_k})",
        xaxis_title="Embedding Axis 1",
        yaxis_title="Embedding Axis 2",
        template="plotly_white",
        legend={"orientation": "h", "y": -0.15},
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return fig


def build_embedding_figure_3d(
    query: str, items: list[MenuItem], retriever: TinyRetriever, top_k: int = 6
) -> go.Figure:
    vocab, matrix = _build_tfidf_vectors(items, retriever)
    q_vec = _query_vector(query, vocab, retriever)
    item_xyz, query_xyz = _project_nd(matrix, q_vec, dims=3)

    retrieved = retriever.retrieve(query, top_k=top_k)
    retrieved_ids = {(r.section, r.name, r.price) for r in retrieved}

    section_colors = _section_color_map(items)
    by_section: dict[str, dict[str, list]] = {}
    hit_x, hit_y, hit_z, hit_text = [], [], [], []

    for idx, item in enumerate(items):
        payload = f"{item.name}<br>{item.section}<br>{item.price}"
        point_x = float(item_xyz[idx, 0])
        point_y = float(item_xyz[idx, 1])
        point_z = float(item_xyz[idx, 2])
        target = (item.section, item.name, item.price)

        if item.section not in by_section:
            by_section[item.section] = {"x": [], "y": [], "z": [], "text": []}
        by_section[item.section]["x"].append(point_x)
        by_section[item.section]["y"].append(point_y)
        by_section[item.section]["z"].append(point_z)
        by_section[item.section]["text"].append(payload)

        if target in retrieved_ids:
            hit_x.append(point_x)
            hit_y.append(point_y)
            hit_z.append(point_z)
            hit_text.append(payload)

    fig = go.Figure()
    for section, payload in by_section.items():
        fig.add_trace(
            go.Scatter3d(
                x=payload["x"],
                y=payload["y"],
                z=payload["z"],
                mode="markers",
                name=section,
                marker={"size": 3, "opacity": 0.55, "color": section_colors[section]},
                text=payload["text"],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=hit_x,
            y=hit_y,
            z=hit_z,
            mode="markers",
            name="Retrieved nodes",
            marker={"size": 6, "opacity": 1.0, "color": "#111111", "symbol": "diamond"},
            text=hit_text,
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[float(query_xyz[0])],
            y=[float(query_xyz[1])],
            z=[float(query_xyz[2])],
            mode="markers",
            name="Your question",
            marker={"size": 8, "color": "#54A24B", "symbol": "diamond"},
            text=[query],
            hovertemplate="Query: %{text}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"3D Embedding Explorer (Top-k retrieved: {top_k})",
        template="plotly_white",
        scene={
            "xaxis_title": "Axis 1",
            "yaxis_title": "Axis 2",
            "zaxis_title": "Axis 3",
        },
        legend={"orientation": "h", "y": -0.12},
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return fig


def build_pyvis_network_html(
    query: str, items: list[MenuItem], retriever: TinyRetriever, top_k: int = 6
) -> str:
    net = Network(height="520px", width="100%", bgcolor="#ffffff", font_color="#222222")
    section_colors = _section_color_map(items)

    query_id = "query-node"
    net.add_node(query_id, label="Your Question", title=query, color="#54A24B", size=28)

    retrieved = retriever.retrieve(query, top_k=top_k)
    retrieved_ids = {(r.section, r.name, r.price) for r in retrieved}

    for idx, item in enumerate(items):
        node_id = f"item-{idx}"
        is_retrieved = (item.section, item.name, item.price) in retrieved_ids
        size = 16 if is_retrieved else 10
        border = "#111111" if is_retrieved else section_colors[item.section]
        title = f"{item.name}<br>{item.section}<br>{item.price}"
        net.add_node(
            node_id,
            label=item.name[:28],
            title=title,
            color={"background": section_colors[item.section], "border": border},
            size=size,
        )
        if is_retrieved:
            net.add_edge(query_id, node_id, color="#111111", width=2)

    net.force_atlas_2based(gravity=-45, central_gravity=0.006, spring_length=120)
    net.show_buttons(filter_=["physics"])

    return net.generate_html(name="embedding_network.html", notebook=False)
