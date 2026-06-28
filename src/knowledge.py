"""Knowledge extraction from the literature via Retrieval-Augmented Generation.

PDFs are parsed with LlamaParse (cached to disk), indexed with LlamaIndex, and
queried with a reranked vector retriever. The public entry point is
`build_retriever`, which returns a zero-argument callable that runs the
configured retrieval query and returns the retrieved text.
"""

import os
import pickle

from .config_loader import require_env, resolve_path, variable_phrase


def _collect_pdf_files(cfg) -> list[str]:
    """Resolve the list of PDF files to parse.

    Each entry in ``knowledge.pdf_files`` may be a single PDF or a directory
    (every ``*.pdf`` inside it is collected). If ``pdf_files`` is empty or
    omitted, every PDF in ``data/literature/`` is parsed automatically, so you
    can drop new PDFs into that folder without editing the config.
    """
    kcfg = cfg["knowledge"]
    entries = kcfg.get("pdf_files") or ["data/literature"]

    files: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        path = resolve_path(cfg, entry)
        matches = sorted(path.glob("*.pdf")) if path.is_dir() else [path]
        for match in matches:
            key = str(match)
            if key not in seen:
                seen.add(key)
                files.append(key)

    if not files:
        raise RuntimeError(
            f"No PDFs found to parse for knowledge.pdf_files={entries!r}. "
            f"Add PDFs to data/literature/ or list them in config.yaml."
        )
    return files


def _load_or_parse_documents(cfg):
    """Parse the literature PDFs (or load a cached parse from disk)."""
    from llama_parse import LlamaParse

    kcfg = cfg["knowledge"]
    cache_path = resolve_path(cfg, kcfg["parsed_cache"])

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    input_files = _collect_pdf_files(cfg)
    parser = LlamaParse(
        api_key=require_env("LLAMA_CLOUD_API_KEY"),
        result_type="markdown",
        parsing_instruction=kcfg["parsing_instruction"],
    )
    documents = parser.load_data(input_files)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(documents, f)
    return documents


def _build_retrieval_query(cfg) -> str:
    """Fill the {topic}/{inputs}/{output} placeholders in the retrieval query."""
    inputs = cfg["variables"]["inputs"]
    output = cfg["variables"]["output"]
    return cfg["knowledge"]["retrieval_query"].format(
        topic=cfg["topic"],
        inputs=variable_phrase(inputs),
        output=f"{output['name']} ({output['symbol']})",
    )


def build_retriever(cfg):
    """Build the RAG query engine and return a `retrieve()` callable.

    Calling the returned function runs the configured retrieval query against
    the indexed literature and returns the retrieved text as a string.
    """
    from llama_index.core import Settings, VectorStoreIndex
    from llama_index.core.node_parser import MarkdownElementNodeParser
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI
    from llama_index.postprocessor.flag_embedding_reranker import (
        FlagEmbeddingReranker,
    )

    openai_key = require_env("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = openai_key
    mcfg = cfg["model"]
    kcfg = cfg["knowledge"]

    documents = _load_or_parse_documents(cfg)

    embed_model = OpenAIEmbedding(model=mcfg["embedding_model"], api_key=openai_key)
    llm = OpenAI(model=mcfg["llm_model"], api_key=openai_key)
    Settings.embed_model = embed_model
    Settings.llm = llm

    node_parser = MarkdownElementNodeParser(llm=llm, num_workers=8)
    nodes = node_parser.get_nodes_from_documents(documents, progress=False)
    base_nodes, objects = node_parser.get_nodes_and_objects(nodes)

    recursive_index = VectorStoreIndex(nodes=base_nodes + objects)

    reranker = FlagEmbeddingReranker(
        top_n=kcfg["reranker_top_n"],
        model=kcfg["reranker_model"],
    )
    query_engine = recursive_index.as_query_engine(
        similarity_top_k=kcfg["similarity_top_k"],
        node_postprocessors=[reranker],
        verbose=False,
    )

    query = _build_retrieval_query(cfg)

    def retrieve() -> str:
        """Run the configured retrieval query and return the retrieved text."""
        response = query_engine.query(query)
        return response.response + "\n\n"

    return retrieve
