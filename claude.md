# Indian Law RAG System — Claude Code Guide

## What this project is
RAG system for Indian law. Users ask legal questions in English, Hindi, or Marathi. Deployed on AWS EC2, FAISS index stored in S3.

## Architecture
Query -> language detection -> translate to English -> BM25 + FAISS hybrid retrieval (RRF) -> GPT-4o sector prompt -> answer

Embedding model: law-ai/InLegalBERT (768-dim). Not trained for retrieval — consider BAAI/bge-base-en.
Retrieval: BM25 weight 2x, dense 1x, top_k=10. FAISS IndexFlatIP (cosine). 2223 vectors, 4 sectors.

## Sectors
- traffic: Motor Vehicles Act, Maharashtra MV Rules, traffic fines PDF
- criminal_law: BNS 2023, BNSS 2023
- matrimonial: Hindu Marriage Act, PWDVA, Dowry Prohibition Act (HMA sector renamed from hindu_marriage_laws)
- rental_law: Maharashtra Rent Control Act 1999

## Key files
- pipeline/build_index.py: builds FAISS index + metadata.json
- pipeline/cleaners/: one cleaner per document; parse() returns chunk dicts
- app/retriever.py: BM25+FAISS+RRF; _NORM dict normalisation; index_info() for health
- app/generator.py: sector prompts; GENERIC_SYSTEM_PROMPT has out-of-domain rejection rule
- app/main.py: FastAPI; /health returns index_built_at, index_source, uptime, chunks_by_sector
- app/index_loader.py: S3 upload/download; no-op if S3_INDEX_BUCKET not set
- deploy.sh: build Docker image, push ECR, restart EC2 via SSH

## Environment variables
- OPENAI_API_KEY: local .env + passed to docker run
- EMBED_MODEL: law-ai/InLegalBERT (must match index build)
- INDEX_DIR: ./index
- S3_INDEX_BUCKET: EC2 docker run only — not in local .env
- AWS_DEFAULT_REGION: eu-north-1

## Deployment
1. Run build_index.py from terminal (NOT Claude Code — AppControl blocks DLL loading on Windows)
2. Upload index: set S3_INDEX_BUCKET and call upload_index_to_s3('./index')
3. Run ./deploy.sh

SSH key: .key1.pem gitignored. Copy to ~/.ssh/rag-key.pem, fix with icacls.
EC2: ubuntu@54.209.6.232, region eu-north-1

View logs:
  ssh -i ~/.ssh/rag-key.pem ubuntu@54.209.6.232 'docker logs rag-indian-law -f'
CloudWatch: /rag-indian-law log group, ec2-container stream (needs CloudWatchLogsFullAccess on IAM role)
Verify: GET /health — check index_built_at and index_source

## Known issues
- InLegalBERT vocabulary gaps (red light vs Jumping signal) — mitigated by BM25 2x, 4x title boost, _NORM, _TITLE_SYNONYMS
- langdetect misidentifies short English queries (e.g. as Danish) — non-critical
- build_index.py blocked by Windows AppControl from Claude Code — use terminal
- Low-confidence guard (score < 0.008) + strict system prompt handles out-of-domain queries

## Run locally
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
