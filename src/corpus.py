from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


def load_default_corpus() -> list[Document]:
    """Return a small default corpus for local RAG practice.

    Students can replace or extend this corpus with course-approved sources.
    """
    return [
        Document(
            doc_id="doc-1",
            title="RAG Fundamentals",
            text=(
                "Retrieval augmented generation combines retrieval with text generation. "
                "The model should answer using retrieved evidence rather than unsupported assumptions."
            ),
        ),
        Document(
            doc_id="doc-2",
            title="Chunking and Metadata",
            text=(
                "Chunking splits documents into smaller parts. Good chunk size and overlap improve retrieval quality. "
                "Metadata helps filtering and source traceability."
            ),
        ),
        Document(
            doc_id="doc-3",
            title="Refusal Behavior",
            text=(
                "If retrieved context does not support a claim, the system should refuse and ask for better evidence. "
                "This reduces hallucinations and increases trust."
            ),
        ),
    ]


def load_training_rag_corpus() -> list[Document]:
    """Return a corpus profile inspired by training_rag notebook examples.

    This profile emphasizes citations, source metadata, and evaluation framing.
    """
    return [
        Document(
            doc_id="tr-doc-1",
            title="RAG with Citations",
            text=(
                "RAG with citations should retrieve relevant documents with source metadata and include references in the answer. "
                "Citations improve transparency and trust because users can verify evidence."
            ),
        ),
        Document(
            doc_id="tr-doc-2",
            title="Structured Citation Output",
            text=(
                "A cited answer can include answer text and list of source titles. "
                "A common format is an answer body followed by a References section."
            ),
        ),
        Document(
            doc_id="tr-doc-3",
            title="LLM-as-a-Judge Evaluation",
            text=(
                "LLM-as-a-judge can evaluate answer quality, citation correctness, and grounding behavior at scale. "
                "It is useful but introduces bias, so metrics should be interpreted carefully."
            ),
        ),
        Document(
            doc_id="tr-doc-4",
            title="Refusal and Grounding",
            text=(
                "When retrieved evidence is missing or irrelevant, the system should refuse unsupported claims. "
                "This behavior reduces hallucinations and aligns with safe RAG practice."
            ),
        ),
    ]


def list_corpus_profiles() -> list[str]:
    return ["default", "training-rag"]


def load_corpus(profile: str = "default") -> list[Document]:
    if profile == "default":
        return load_default_corpus()
    if profile == "training-rag":
        return load_training_rag_corpus()
    raise ValueError(f"Unknown corpus profile: {profile}")
