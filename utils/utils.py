"""Shared notebook helpers."""


def show_graph(graph, xray=False):
    """Display a LangGraph mermaid diagram with ASCII fallback."""
    from IPython.display import Image
    try:
        return Image(graph.get_graph(xray=xray).draw_mermaid_png())
    except Exception as e:
        print(f"Image rendering failed: {e}")
        print("\nFalling back to ASCII:\n")
        print(graph.get_graph(xray=xray).draw_ascii())
        return None
