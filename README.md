# The Prince Machiavelli RAG

Retrieval augmented generation over Niccolò Machiavelli's *The Prince*
(Project Gutenberg #1232, W. K. Marriott translation).

This project starts from the Week 6 RAG homework pipeline
(`corpus → chunking → retrieval → rag → cli`): Python 3.9, standard library only,
extractive answers with chunk citations, and refusal when evidence is missing.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./scripts/run_tests.sh
```
