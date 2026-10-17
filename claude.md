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

SSH key: .key2.pem gitignored. Copy to ~/.ssh/rag-key-new.pem, fix with icacls.
EC2: ubuntu@13.49.33.25 (Elastic IP, alloc eipalloc-00f6a12b1017c9b09), region eu-north-1, instance type t3.small
(CpuCredits=unlimited), 20GB gp3 root volume, 1GB swap at /swapfile.
AWS account: 550842783429 (migrated from the original free-tier account 2026-10-15; use `aws --profile rag-new`)
S3 bucket: rag-indian-law-index-550842783429 (renamed — bucket names are globally unique, old name was taken)

Note: t3.micro (1GB RAM, default burstable credits, no swap) could NOT run this app — torch+transformers+InLegalBERT+
FAISS+BM25 exhausted CPU credits during model load and then got so throttled that even SSH stopped responding; a
reboot didn't recover it. Resized to t3.small (2GB RAM) with CpuCredits=unlimited, added 1GB swap, which fixed it.
Don't downsize below t3.small. Elastic IP is attached so the public IP no longer changes across stop/start/resize —
previously it did (bit the migration once already), so if DNS ever needs updating again, something detached the EIP.

HTTPS/domain: nginx + certbot installed directly on the EC2 box (NOT in this repo, not in Docker — reinstall manually
on any future box: `apt install nginx certbot python3-certbot-nginx`, reverse-proxy 80 -> 127.0.0.1:8000, then
`certbot --nginx -d shastrashaw.online -d www.shastrashaw.online --redirect`). DNS (at the registrar, not AWS/Route53):
A records for both `@`/root and `www` -> the Elastic IP above. Cert auto-renews via certbot's systemd timer.
We don't actually know how the OLD box did HTTPS — this setup was done fresh during the AWS account migration,
2026-10-15, with no way to inspect the old instance's config.

View logs:
  ssh -i ~/.ssh/rag-key-new.pem ubuntu@13.49.33.25 'docker logs rag-indian-law -f'
CloudWatch: /rag-indian-law log group, ec2-container stream (needs CloudWatchLogsFullAccess on IAM role)
Verify: GET /health — check index_built_at and index_source

## Security
Full-codebase audit done 2026-10-15 (code review + live checks against the deployed site). No injection/XSS/CORS/
SSRF/auth-bypass findings: app/memory.py uses parameterized SQLModel queries (no string-interpolated SQL), the
frontend (app/static/js/chat.js) renders model output through marked.parse() -> DOMPurify.sanitize() with a strict
allowlist plus an `esc()` helper for everything else, no CORSMiddleware is configured at all, no subprocess/eval/
pickle/yaml.load anywhere under app/, session_id is a non-guessable server-generated uuid4 used only as a memory key
(no access-control semantics to bypass).

Two real gaps found via live testing (not code review) and fixed:
- `/docs`, `/redoc`, `/openapi.json` were publicly browsable (free API recon for an unauthenticated public app) —
  now off by default in app/main.py; set `ENABLE_API_DOCS=1` to re-enable locally for dev.
- No security headers and nginx leaked its exact version — added HSTS, X-Content-Type-Options, X-Frame-Options,
  Referrer-Policy via `add_header` in /etc/nginx/sites-available/shastrashaw.online (NOT in this repo, box-level —
  redo on any future box), and uncommented `server_tokens off;` in /etc/nginx/nginx.conf.

Re-verify after any redeploy: `curl -s -o /dev/null -w "%{http_code}\n" https://www.shastrashaw.online/docs` should
be 404, not 200 (docker build/push/deploy can be slow enough that a session interruption mid-deploy leaves the OLD
image still running — check `docker ps` uptime on the box against the last deploy time if unsure).

## Reliability / ops (added 2026-10-17, box-level, NOT in this repo — redo on any future box)
Goal: match a funded-startup baseline on a single server, no horizontal scaling. All AWS pieces live in account
550842783429, region eu-north-1 (profile `rag-new`).

- **nginx rate limiting**: `limit_req_zone` in /etc/nginx/nginx.conf (10 req/min/IP, zone `api_limit`), applied in
  the site's `location /` block with `burst=15 nodelay`. Protects the OpenAI bill from being run up by abuse/bots —
  this is the single highest-value protection given every `/query` call is a metered GPT-4o call. Returns 429 when
  exceeded. Verify: fire >15 rapid requests at /health, should see 200s then 429s.
- **fail2ban**: installed, sshd jail enabled, default config (bans after repeated failed SSH logins). Check:
  `sudo fail2ban-client status sshd`.
- **Daily EBS snapshots**: AWS DLM lifecycle policy `policy-06e179eec8769091b`, targets the volume tagged
  `Backup=daily`, runs 03:00 UTC, 7-day retention. This is the disaster-recovery answer for everything that's
  box-level and NOT in git (nginx config, certbot certs, fail2ban, this whole Reliability section's setup) —
  if the instance is lost, restore from the latest snapshot rather than rebuilding by hand from CLAUDE.md notes.
- **Chat-memory DB backup**: /usr/local/bin/backup-chat-db.sh (cron, daily 03:17 UTC) does a sqlite3 `.backup` of
  /home/ubuntu/data/chat_history.db and uploads to s3://rag-indian-law-index-550842783429/backups/ (14-day S3
  lifecycle expiry). EC2 role has a scoped `s3-backup-write` inline policy for PutObject under `backups/*` only.
- **Monitoring**: CloudWatch agent installed (`CWAgent` namespace, dimension is `host=ip-172-31-45-191`, NOT
  `InstanceId` — that tripped up alarm setup once already), reporting `mem_used_percent` and `disk_used_percent`.
  Four CloudWatch alarms (`rag-indian-law-status-check-failed`, `-high-cpu`, `-high-memory`, `-low-disk`) all notify
  SNS topic `rag-indian-law-alerts` -> email. **The email subscription must be manually confirmed** (AWS sends a
  confirmation link) or alarms fire silently with no notification — check `aws sns list-subscriptions-by-topic`
  isn't stuck on `PendingConfirmation`.
- **Still not done**: Cloudflare (WAF + DDoS protection + CDN) — needs the domain's nameservers pointed at
  Cloudflare, which is a registrar-side change the user has to make; not done as of 2026-10-17.

## Known issues
- InLegalBERT vocabulary gaps (red light vs Jumping signal) — mitigated by BM25 2x, 4x title boost, _NORM, _TITLE_SYNONYMS
- langdetect misidentifies short English queries (e.g. as Danish) — non-critical
- build_index.py blocked by Windows AppControl from Claude Code — use terminal
- Low-confidence guard (score < 0.008) + strict system prompt handles out-of-domain queries

## Run locally
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
