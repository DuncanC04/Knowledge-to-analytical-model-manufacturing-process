"""Local literature retrieval for equation generation.

PDFs are parsed locally with Docling, cached as Markdown, indexed with
LlamaIndex using local HuggingFace embeddings, and reranked with a local BGE
reranker. No LlamaParse or Llama Cloud API key is required.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config_loader import resolve_path, variable_phrase

REFERENCE_HEADING = re.compile(
    r"(?im)^[ \t]{0,3}(?:#{1,6}[ \t]+)?"
    r"(?:references|bibliography|works cited|reference list)"
    r"[ \t]*:?[ \t]*$"
)
PARAMETRIC_EVIDENCE = re.compile(
    r"(?i)(?:regression|response\s+surface|polynomial|power[- ]law|parametric|"
    r"coded\s+variables?|coefficient|model\s+equation|fitted\s+equation|"
    r"\b[abcd]\s*[-:]\s*(?:water|pulse|frequency|scanning)|"
    r"\b[abcd]\s*[²2]\b|\bx\s*[1234567890]\b|\by\s*=)"
)


@dataclass(frozen=True)
class LocalRetrievalConfig:
    """Settings for the local parser/index/reranker pipeline."""

    embedding_model: str = "BAAI/bge-large-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-large"
    reranker_top_n: int = 10
    retrieval_top_k: int = 20
    chunk_size: int = 800
    chunk_overlap: int = 120
    remove_reference_section: bool = True


class ParametricReranker:
    """Apply semantic reranking, then prefer passages with equation evidence."""

    def __init__(self, reranker: Any, top_n: int, bonus: float = 0.25):
        self.reranker = reranker
        self.top_n = top_n
        self.bonus = bonus

    def postprocess_nodes(self, nodes: list[Any], query_str: str = "") -> list[Any]:
        ranked_nodes = self.reranker.postprocess_nodes(nodes, query_str=query_str)
        for node in ranked_nodes:
            if PARAMETRIC_EVIDENCE.search(node.node.get_content()):
                node.score = (node.score or 0.0) + self.bonus
        ranked_nodes.sort(key=lambda node: node.score or 0.0, reverse=True)
        return ranked_nodes[: self.top_n]


def _collect_pdf_files(cfg) -> list[Path]:
    """Resolve the list of PDF files to parse."""
    kcfg = cfg["knowledge"]
    entries = kcfg.get("pdf_files") or ["data/literature"]

    files: list[Path] = []
    seen: set[Path] = set()
    for entry in entries:
        path = resolve_path(cfg, entry)
        matches = sorted(path.glob("*.pdf")) if path.is_dir() else [path]
        for match in matches:
            resolved = match.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(match)

    if not files:
        raise RuntimeError(
            f"No PDFs found to parse for knowledge.pdf_files={entries!r}. "
            f"Add PDFs to data/literature/ or list them in config.yaml."
        )
    return files


def _local_config(cfg) -> LocalRetrievalConfig:
    """Read local retrieval settings, falling back to conservative defaults."""
    kcfg = cfg["knowledge"]
    return LocalRetrievalConfig(
        embedding_model=kcfg.get("embedding_model", "BAAI/bge-large-en-v1.5"),
        reranker_model=kcfg.get("reranker_model", "BAAI/bge-reranker-large"),
        reranker_top_n=kcfg.get("reranker_top_n", 10),
        retrieval_top_k=max(
            kcfg.get("similarity_top_k", 20),
            kcfg.get("reranker_top_n", 10),
        ),
        chunk_size=kcfg.get("chunk_size", 800),
        chunk_overlap=kcfg.get("chunk_overlap", 120),
        remove_reference_section=kcfg.get("remove_reference_section", True),
    )


def _parsed_cache_dir(cfg) -> Path:
    """Use a folder next to the configured parse cache for Docling Markdown."""
    cache_path = resolve_path(cfg, cfg["knowledge"]["parsed_cache"])
    if cache_path.suffix:
        return cache_path.with_suffix("")
    return cache_path


def _remove_reference_section(markdown_text: str) -> str:
    """Remove a bibliography beginning at a standalone Markdown heading."""
    match = REFERENCE_HEADING.search(markdown_text)
    return markdown_text[: match.start()].rstrip() if match else markdown_text


def _compute_device() -> str:
    """Return the accelerator used by local embedding and reranking models."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _create_converter():
    """Create the local Docling PDF converter."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options.device = _compute_device()
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        },
    )


def _load_markdown(pdf_path: Path, converter: Any, cache_dir: Path) -> str:
    """Load cached Markdown or convert one PDF and update its cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{pdf_path.stem}.json"
    source_stat = pdf_path.stat()

    if cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as cache_file:
                cached_data = json.load(cache_file)
            if (
                cached_data.get("source_size") == source_stat.st_size
                and cached_data.get("source_mtime_ns") == source_stat.st_mtime_ns
            ):
                return cached_data["markdown"]
        except (KeyError, OSError, json.JSONDecodeError):
            cache_path.unlink(missing_ok=True)

    result = converter.convert(str(pdf_path))
    markdown_text = result.document.export_to_markdown()
    with cache_path.open("w", encoding="utf-8") as cache_file:
        json.dump(
            {
                "source_file": pdf_path.name,
                "source_size": source_stat.st_size,
                "source_mtime_ns": source_stat.st_mtime_ns,
                "markdown": markdown_text,
            },
            cache_file,
            ensure_ascii=False,
            indent=2,
        )
    return markdown_text


def _load_documents(cfg, local_cfg: LocalRetrievalConfig):
    """Parse PDFs into LlamaIndex documents, using the local parse cache."""
    from llama_index.core import Document

    converter = _create_converter()
    cache_dir = _parsed_cache_dir(cfg)
    documents = []
    for pdf_path in _collect_pdf_files(cfg):
        markdown_text = _load_markdown(pdf_path, converter, cache_dir)
        if local_cfg.remove_reference_section:
            markdown_text = _remove_reference_section(markdown_text)
        documents.append(
            Document(
                text=markdown_text,
                metadata={"file_name": pdf_path.name, "file_path": str(pdf_path)},
            )
        )

    if not documents:
        raise RuntimeError("No documents were successfully loaded.")
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


def _configure_local_models(local_cfg: LocalRetrievalConfig) -> None:
    """Configure local embeddings and chunking for LlamaIndex."""
    from llama_index.core import Settings
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=local_cfg.embedding_model,
        device=_compute_device(),
    )
    Settings.text_splitter = SentenceSplitter(
        chunk_size=local_cfg.chunk_size,
        chunk_overlap=local_cfg.chunk_overlap,
    )


def _build_reranker(local_cfg: LocalRetrievalConfig) -> ParametricReranker:
    """Create the local FlagEmbedding reranker."""
    import torch
    from llama_index.postprocessor.flag_embedding_reranker import (
        FlagEmbeddingReranker,
    )

    reranker = FlagEmbeddingReranker(
        top_n=local_cfg.retrieval_top_k,
        model=local_cfg.reranker_model,
        use_fp16=torch.cuda.is_available(),
    )
    return ParametricReranker(reranker, top_n=local_cfg.reranker_top_n)


def _format_nodes(nodes: list[Any]) -> str:
    """Format reranked retrieved nodes as grounded context text."""
    chunks = []
    for index, node in enumerate(nodes, start=1):
        metadata = node.node.metadata or {}
        source = metadata.get("file_name", "unknown source")
        score = 0.0 if node.score is None else node.score
        chunks.append(
            f"[{index}] source={source} score={score:.4f}\n"
            f"{node.node.get_content()}"
        )
    return "\n\n".join(chunks) + "\n\n"


def build_retriever(cfg):
    """Build the local retrieval pipeline and return a `retrieve()` callable.

    Calling the returned function runs the configured retrieval query against
    locally parsed and locally embedded literature, then returns reranked text.
    """
    from llama_index.core import VectorStoreIndex

    local_cfg = _local_config(cfg)
    if local_cfg.chunk_overlap >= local_cfg.chunk_size:
        raise ValueError("knowledge.chunk_overlap must be smaller than chunk_size.")

    _configure_local_models(local_cfg)
    documents = _load_documents(cfg, local_cfg)
    index = VectorStoreIndex.from_documents(documents)
    retriever = index.as_retriever(similarity_top_k=local_cfg.retrieval_top_k)
    reranker = _build_reranker(local_cfg)
    query = _build_retrieval_query(cfg)

    def retrieve() -> str:
        """Run local retrieval and return reranked text chunks."""
        nodes = retriever.retrieve(query)
        reranked_nodes = reranker.postprocess_nodes(nodes, query_str=query)
        return _format_nodes(reranked_nodes)

    return retrieve
