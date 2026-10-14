import argparse

from .chunking import (
    CHUNKING_STRATEGIES,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    DEFAULT_SENTENCE_OVERLAP,
    build_chunks,
)
from .corpus import list_corpus_profiles, load_corpus
from .rag import build_grounded_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 6 RAG homework starter")
    parser.add_argument("--question", required=True, help="Question to ask over the local corpus")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size in characters")
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP, help="Chunk overlap in characters")
    parser.add_argument(
        "--strategy",
        default="char",
        choices=CHUNKING_STRATEGIES,
        help="Chunking strategy: char (fixed character windows) or sentence (whole sentences)",
    )
    parser.add_argument(
        "--sentence-overlap",
        type=int,
        default=DEFAULT_SENTENCE_OVERLAP,
        help="Sentences repeated between chunks (sentence strategy only)",
    )
    parser.add_argument("--stem", action="store_true", help="Match word forms with light stemming (helps = help)")
    parser.add_argument("--top-k", type=int, default=2, help="Number of retrieved chunks")
    parser.add_argument(
        "--corpus-profile",
        default="default",
        choices=list_corpus_profiles(),
        help="Corpus profile: default or training-rag",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus = load_corpus(args.corpus_profile)
    chunks = build_chunks(
        corpus,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        strategy=args.strategy,
        sentence_overlap=args.sentence_overlap,
    )
    result = build_grounded_answer(args.question, chunks, top_k=args.top_k, use_stemming=args.stem)

    print("Question:")
    print(args.question)
    print("\nCorpus profile:", args.corpus_profile)
    if args.strategy == "sentence":
        print(
            f"Chunking: strategy=sentence, chunk_size={args.chunk_size} (characters), "
            f"sentence_overlap={args.sentence_overlap} (sentences)"
        )
    else:
        print(f"Chunking: chunk_size={args.chunk_size}, overlap={args.overlap} (characters)")
    if args.stem:
        print("Matching: stemming=on")
    print(f"Top-k: {args.top_k}; corpus chunks: {len(chunks)}")
    print("\nRetrieved context (descending score):")
    if not result.retrieved:
        print("None")
    for rank, item in enumerate(result.retrieved, start=1):
        print(f"{rank}. {item.chunk.chunk_id} | score={item.score:.6f} | {item.chunk.title}")
        print(item.chunk.text)
    print("\nAnswer:")
    print(result.answer)
    print("\nRefused:", result.refused)
    print("Citations:", ", ".join(result.citations) if result.citations else "None")


if __name__ == "__main__":
    main()
