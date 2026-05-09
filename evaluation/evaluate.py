"""
evaluation/evaluate.py — Evaluation suite for the Indian Law RAG system.

Computes three tiers of metrics:
  1. Retrieval   — Hit@K, Recall@K, MRR, NDCG@K
  2. Generation  — BLEU, ROUGE-1/2/L, BERTScore
  3. End-to-end  — RAGAS (Faithfulness, Answer Relevancy,
                          Context Precision, Context Recall)

Usage:
    python evaluation/evaluate.py \\
        --index_dir  ./index \\
        --dataset    ./evaluation/gold_dataset.json \\
        --out_dir    ./evaluation/results \\
        [--top_k 5] \\
        [--skip_ragas] \\
        [--skip_generation]

Outputs written to --out_dir:
    retrieval_results.csv
    generation_results.csv
    ragas_results.csv        (unless --skip_ragas)
    summary.json
    plots/
"""

import os
import sys
import json
from dotenv import load_dotenv
load_dotenv()
import math
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser(description="RAG evaluation suite for Indian Law system")
    p.add_argument("--index_dir",       default="./index",
                   help="Path to FAISS index directory")
    p.add_argument("--dataset",         default="./evaluation/gold_dataset.json",
                   help="Path to gold dataset JSON")
    p.add_argument("--out_dir",         default="./evaluation/results",
                   help="Directory to save results and plots")
    p.add_argument("--top_k",           type=int, default=5,
                   help="Number of chunks to retrieve per query (default: 5)")
    p.add_argument("--embed_model",     default="law-ai/InLegalBERT",
                   help="Sentence-transformer model used to build the index")
    p.add_argument("--skip_generation", action="store_true",
                   help="Skip GPT-4 generation (compute retrieval metrics only)")
    p.add_argument("--skip_ragas",      action="store_true",
                   help="Skip RAGAS evaluation")
    p.add_argument("--openai_model",    default="gpt-4o",
                   help="OpenAI model to use for generation (default: gpt-4o)")
    return p.parse_args()


def is_relevant(chunk: dict, expected_sources: list, expected_section: str) -> bool:
    """
    A chunk is relevant if its source matches any expected source keyword,
    or its section number matches the expected section.
    """
    chunk_source  = chunk.get("source", "").lower()
    chunk_section = str(chunk.get("section_number", "")).strip()

    for src in expected_sources:
        keywords = src.replace("_", " ").lower().split()
        if any(kw in chunk_source for kw in keywords if len(kw) > 3):
            return True

    if expected_section and chunk_section == str(expected_section).strip():
        return True

    return False


def hit_at_k(results: list, expected_sources: list, expected_section: str) -> int:
    """1 if at least one relevant chunk is in the results, else 0."""
    return int(any(is_relevant(r, expected_sources, expected_section) for r in results))


def recall_at_k(results: list, expected_sources: list, expected_section: str) -> float:
    """Fraction of retrieved chunks that are relevant."""
    relevant = sum(is_relevant(r, expected_sources, expected_section) for r in results)
    return relevant / max(len(results), 1)


def reciprocal_rank(results: list, expected_sources: list, expected_section: str) -> float:
    """1 / rank of first relevant result. 0 if none found."""
    for i, r in enumerate(results, start=1):
        if is_relevant(r, expected_sources, expected_section):
            return 1.0 / i
    return 0.0


def ndcg_at_k(results: list, expected_sources: list, expected_section: str) -> float:
    """Normalized Discounted Cumulative Gain at K."""
    gains = [
        1.0 if is_relevant(r, expected_sources, expected_section) else 0.0
        for r in results
    ]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    n_relevant   = int(sum(gains))
    ideal_gains  = [1.0] * n_relevant + [0.0] * (len(gains) - n_relevant)
    idcg         = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_gains))
    return dcg / idcg if idcg > 0 else 0.0


def compute_retrieval_metrics(item: dict, results: list) -> dict:
    src = item["expected_sources"]
    sec = item.get("expected_section", "")
    return {
        "id"            : item["id"],
        "sector"        : item["sector"],
        "question"      : item["question"][:80],
        "hit_at_k"      : hit_at_k(results, src, sec),
        "recall_at_k"   : recall_at_k(results, src, sec),
        "mrr"           : reciprocal_rank(results, src, sec),
        "ndcg_at_k"     : ndcg_at_k(results, src, sec),
        "top_score"     : results[0]["score"] if results else 0.0,
        "num_retrieved" : len(results),
    }


def compute_bleu(reference: str, hypothesis: str) -> float:
    """Sentence BLEU-4 using NLTK (smoothed)."""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    ref_tokens  = reference.lower().split()
    hyp_tokens  = hypothesis.lower().split()
    smoothie    = SmoothingFunction().method4
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)


def compute_rouge(reference: str, hypothesis: str) -> dict:
    """ROUGE-1, ROUGE-2, ROUGE-L F1 scores."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


def compute_bert_score(references: list, hypotheses: list) -> dict:
    """BERTScore F1 using microsoft/deberta-xlarge-mnli."""
    from bert_score import score as bert_score_fn
    print("  Computing BERTScore (this may take a minute on first run)...")
    P, R, F1 = bert_score_fn(
        hypotheses, references,
        model_type="microsoft/deberta-xlarge-mnli",
        verbose=False,
        device="cpu",
    )
    return {
        "bertscore_precision": P.mean().item(),
        "bertscore_recall"   : R.mean().item(),
        "bertscore_f1"       : F1.mean().item(),
        "bertscore_per_item" : F1.tolist(),
    }


def _configure_ragas_metrics(metrics: list, model: str = "gpt-4o"):
    """
    Wire each RAGAS metric to a configured LLM and embeddings wrapper.
    Avoids the default gpt-4o-mini token limit issues with long faithfulness outputs.
    """
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        ragas_llm = LangchainLLMWrapper(ChatOpenAI(model=model, max_tokens=4096))
        ragas_emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

        for metric in metrics:
            if hasattr(metric, "llm"):
                metric.llm = ragas_llm
            if hasattr(metric, "embeddings"):
                metric.embeddings = ragas_emb

        print(f"  RAGAS configured with {model} (max_tokens=4096)")
        return True
    except Exception as e:
        print(f"  Could not configure RAGAS LLM/embeddings: {e}")
        print("     Install: pip install langchain-openai")
        return False


def run_ragas(ragas_rows: list, model: str = "gpt-4o") -> dict:
    """Run RAGAS end-to-end evaluation. Supports RAGAS 0.2.x with 0.1.x fallback."""
    try:
        from datasets import Dataset
        ds = Dataset.from_list(ragas_rows)
    except ImportError:
        print("  'datasets' not installed — skipping RAGAS. pip install datasets")
        return {}

    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
    except ImportError:
        print("  'ragas' not installed — skipping. pip install ragas")
        return {}

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    _configure_ragas_metrics(metrics, model=model)

    try:
        print("  Using RAGAS 0.2.x API...")
        result = evaluate(ds, metrics=metrics)
        df = result.to_pandas()
        return {
            "faithfulness"      : float(df["faithfulness"].mean()),
            "answer_relevancy"  : float(df["answer_relevancy"].mean()),
            "context_precision" : float(df["context_precision"].mean()),
            "context_recall"    : float(df["context_recall"].mean()),
            "per_item"          : df.to_dict(orient="records"),
        }
    except Exception as e_new:
        print(f"  RAGAS 0.2.x failed ({e_new}), trying 0.1.x API...")
        try:
            result = evaluate(ds, metrics=metrics)
            return {
                "faithfulness"      : result["faithfulness"],
                "answer_relevancy"  : result["answer_relevancy"],
                "context_precision" : result["context_precision"],
                "context_recall"    : result["context_recall"],
                "per_item"          : [],
            }
        except Exception as e_old:
            print(f"  RAGAS evaluation failed: {e_old}")
            return {}


def plot_retrieval_by_sector(ret_rows: list, out_dir: Path):
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(ret_rows)
    sectors = df["sector"].unique()
    metrics = ["hit_at_k", "recall_at_k", "mrr", "ndcg_at_k"]
    labels  = ["Hit@K", "Recall@K", "MRR", "NDCG@K"]

    sector_means = df.groupby("sector")[metrics].mean()

    x    = np.arange(len(sectors))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = [sector_means.loc[s, metric] if s in sector_means.index else 0 for s in sectors]
        ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([s.replace("_", "\n") for s in sectors], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Retrieval Metrics by Sector (top_k={ret_rows[0].get('num_retrieved','?')})")
    ax.legend(loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = out_dir / "retrieval_by_sector.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_generation_by_sector(gen_rows: list, out_dir: Path):
    import matplotlib.pyplot as plt
    import pandas as pd

    df      = pd.DataFrame(gen_rows)
    sectors = df["sector"].unique()
    metrics = ["bleu", "rouge1", "rouge2", "rougeL", "bertscore_f1"]
    labels  = ["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore F1"]

    available = [m for m in metrics if m in df.columns]
    avail_labels = [labels[metrics.index(m)] for m in available]

    sector_means = df.groupby("sector")[available].mean()

    x    = np.arange(len(sectors))
    width = 0.14
    fig, ax = plt.subplots(figsize=(14, 6))

    colors = ["#F44336", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    for i, (metric, label, color) in enumerate(zip(available, avail_labels, colors)):
        vals = [sector_means.loc[s, metric] if s in sector_means.index else 0 for s in sectors]
        ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)

    ax.set_xticks(x + width * (len(available) - 1) / 2)
    ax.set_xticklabels([s.replace("_", "\n") for s in sectors], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Generation Metrics by Sector")
    ax.legend(loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = out_dir / "generation_by_sector.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_overall_summary(summary: dict, out_dir: Path):
    import matplotlib.pyplot as plt

    all_metrics = {}

    if "retrieval" in summary:
        r = summary["retrieval"]
        all_metrics.update({
            "Hit@K"     : r.get("hit_at_k_mean", 0),
            "Recall@K"  : r.get("recall_at_k_mean", 0),
            "MRR"       : r.get("mrr_mean", 0),
            "NDCG@K"    : r.get("ndcg_at_k_mean", 0),
        })
    if "generation" in summary:
        g = summary["generation"]
        all_metrics.update({
            "BLEU"          : g.get("bleu_mean", 0),
            "ROUGE-1"       : g.get("rouge1_mean", 0),
            "ROUGE-2"       : g.get("rouge2_mean", 0),
            "ROUGE-L"       : g.get("rougeL_mean", 0),
            "BERTScore F1"  : g.get("bertscore_f1_mean", 0),
        })
    if "ragas" in summary:
        rg = summary["ragas"]
        all_metrics.update({
            "Faithfulness"      : rg.get("faithfulness", 0),
            "Ans. Relevancy"    : rg.get("answer_relevancy", 0),
            "Ctx. Precision"    : rg.get("context_precision", 0),
            "Ctx. Recall"       : rg.get("context_recall", 0),
        })

    if not all_metrics:
        return

    labels = list(all_metrics.keys())
    values = list(all_metrics.values())

    colors = []
    for l in labels:
        if l in ("Hit@K", "Recall@K", "MRR", "NDCG@K"):
            colors.append("#2196F3")
        elif l in ("BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore F1"):
            colors.append("#4CAF50")
        else:
            colors.append("#FF9800")

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.barh(labels, values, color=colors, alpha=0.85, edgecolor="white")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9,
        )

    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Score")
    ax.set_title("Overall System Performance — All Metrics")
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2196F3", label="Retrieval"),
        Patch(facecolor="#4CAF50", label="Generation"),
        Patch(facecolor="#FF9800", label="RAGAS"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    path = out_dir / "overall_summary.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_score_distribution(ret_rows: list, gen_rows: list, out_dir: Path):
    """Box plots showing metric score distributions."""
    import matplotlib.pyplot as plt
    import pandas as pd

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ret_df = pd.DataFrame(ret_rows)
    ret_metrics = ["hit_at_k", "recall_at_k", "mrr", "ndcg_at_k"]
    ret_data = [ret_df[m].tolist() for m in ret_metrics if m in ret_df.columns]
    ret_labels = ["Hit@K", "Recall@K", "MRR", "NDCG@K"]

    bp1 = axes[0].boxplot(ret_data, labels=ret_labels, patch_artist=True)
    for patch, color in zip(bp1["boxes"], ["#2196F3", "#64B5F6", "#1565C0", "#42A5F5"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set_title("Retrieval Metric Distributions")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(-0.05, 1.1)
    axes[0].yaxis.grid(True, linestyle="--", alpha=0.5)

    if gen_rows:
        gen_df = pd.DataFrame(gen_rows)
        gen_metrics = [m for m in ["bleu", "rouge1", "rouge2", "rougeL", "bertscore_f1"]
                       if m in gen_df.columns]
        gen_data   = [gen_df[m].tolist() for m in gen_metrics]
        gen_labels = [m.upper().replace("ROUGEL", "ROUGE-L")
                                .replace("BERTSCORE_F1", "BERTScore\nF1")
                                .replace("ROUGE1", "ROUGE-1")
                                .replace("ROUGE2", "ROUGE-2")
                      for m in gen_metrics]
        bp2 = axes[1].boxplot(gen_data, labels=gen_labels, patch_artist=True)
        colors2 = ["#F44336", "#4CAF50", "#66BB6A", "#81C784", "#9C27B0"]
        for patch, color in zip(bp2["boxes"], colors2):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1].set_title("Generation Metric Distributions")
        axes[1].set_ylabel("Score")
        axes[1].set_ylim(-0.05, 1.1)
        axes[1].yaxis.grid(True, linestyle="--", alpha=0.5)
    else:
        axes[1].set_visible(False)

    plt.tight_layout()
    path = out_dir / "score_distributions.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_heatmap(ret_rows: list, out_dir: Path):
    """Heatmap of retrieval metrics by sector."""
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(ret_rows)
    metrics = ["hit_at_k", "recall_at_k", "mrr", "ndcg_at_k"]
    labels  = ["Hit@K", "Recall@K", "MRR", "NDCG@K"]
    pivot = df.groupby("sector")[metrics].mean()
    pivot.columns = labels

    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.9 + 1)))
    im = ax.imshow(pivot.values, cmap="YlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([s.replace("_", " ").title() for s in pivot.index], fontsize=10)
    ax.set_title("Retrieval Performance Heatmap (by Sector)")

    for i in range(len(pivot.index)):
        for j in range(len(labels)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="black" if val < 0.7 else "white", fontsize=11, fontweight="bold")

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    path = out_dir / "retrieval_heatmap.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def save_csv(rows: list, path: Path):
    import csv
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {path}")


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 65)
    print("  INDIAN LAW RAG — EVALUATION SUITE")
    print("=" * 65)

    with open(args.dataset, encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"\n[Dataset] Loaded {len(dataset)} Q&A pairs from {args.dataset}")

    print(f"\n[Retriever] Loading FAISS index from {args.index_dir} ...")
    from app.retriever import Retriever
    retriever = Retriever(index_dir=args.index_dir, model_name=args.embed_model)
    print(f"  Ready — {retriever.index.ntotal} vectors")

    generator = None
    if not args.skip_generation:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("\n[Generator] OPENAI_API_KEY not set — skipping generation.")
            args.skip_generation = True
        else:
            from app.generator import Generator
            generator = Generator(api_key=api_key, model=args.openai_model)
            print(f"[Generator] Using {args.openai_model}")

    # Tier 1 — Retrieval
    print(f"\n{'─'*65}")
    print(f"[1/3] RETRIEVAL EVALUATION  (top_k={args.top_k})")
    print(f"{'─'*65}")

    retrieval_rows = []
    retrieved_per_item = {}

    for item in dataset:
        results = retriever.search(
            item["question"],
            top_k=args.top_k,
            sector_filter=item["sector"],
        )
        retrieved_per_item[item["id"]] = results
        row = compute_retrieval_metrics(item, results)
        retrieval_rows.append(row)
        status = "OK" if row["hit_at_k"] else "MISS"
        print(f"  [{status}] [{item['id']}] {item['question'][:55]}")
        print(f"       Hit@{args.top_k}={row['hit_at_k']}  "
              f"Recall={row['recall_at_k']:.2f}  "
              f"MRR={row['mrr']:.2f}  "
              f"NDCG={row['ndcg_at_k']:.2f}  "
              f"TopScore={row['top_score']:.3f}")

    ret_means = {
        "hit_at_k_mean"   : np.mean([r["hit_at_k"]    for r in retrieval_rows]),
        "recall_at_k_mean": np.mean([r["recall_at_k"] for r in retrieval_rows]),
        "mrr_mean"        : np.mean([r["mrr"]         for r in retrieval_rows]),
        "ndcg_at_k_mean"  : np.mean([r["ndcg_at_k"]  for r in retrieval_rows]),
        "top_score_mean"  : np.mean([r["top_score"]   for r in retrieval_rows]),
    }

    print(f"\n  Retrieval Summary")
    print(f"  Hit@{args.top_k}     : {ret_means['hit_at_k_mean']:.4f}")
    print(f"  Recall@{args.top_k}  : {ret_means['recall_at_k_mean']:.4f}")
    print(f"  MRR        : {ret_means['mrr_mean']:.4f}")
    print(f"  NDCG@{args.top_k}    : {ret_means['ndcg_at_k_mean']:.4f}")
    print(f"  Avg Sim    : {ret_means['top_score_mean']:.4f}")

    # Tier 2 — Generation
    generation_rows = []
    generated_answers = {}
    gen_means = {}

    if not args.skip_generation:
        print(f"\n{'─'*65}")
        print(f"[2/3] GENERATION EVALUATION  ({args.openai_model})")
        print(f"{'─'*65}")

        print("\n  Generating answers...")
        for item in dataset:
            results = retrieved_per_item[item["id"]]
            try:
                gen_result = generator.generate(
                    question=item["question"],
                    context_chunks=results,
                    sector=item["sector"],
                )
                answer = gen_result["answer"]
            except Exception as e:
                print(f"  Generation failed for {item['id']}: {e}")
                answer = ""
            generated_answers[item["id"]] = answer

        print("\n  Computing BLEU and ROUGE...")
        for item in dataset:
            answer    = generated_answers.get(item["id"], "")
            reference = item["ground_truth"]
            if not answer:
                generation_rows.append({
                    "id": item["id"], "sector": item["sector"],
                    "bleu": 0, "rouge1": 0, "rouge2": 0, "rougeL": 0,
                })
                continue

            bleu_score  = compute_bleu(reference, answer)
            rouge_scores = compute_rouge(reference, answer)

            row = {
                "id"      : item["id"],
                "sector"  : item["sector"],
                "question": item["question"][:80],
                "bleu"    : round(bleu_score, 4),
                **{k: round(v, 4) for k, v in rouge_scores.items()},
            }
            generation_rows.append(row)

        print("\n  Computing BERTScore (batch)...")
        try:
            ids_with_answers = [item["id"] for item in dataset
                                if generated_answers.get(item["id"])]
            refs = [item["ground_truth"] for item in dataset
                    if generated_answers.get(item["id"])]
            hyps = [generated_answers[item["id"]] for item in dataset
                    if generated_answers.get(item["id"])]

            bs = compute_bert_score(refs, hyps)
            bs_per_item = {
                iid: f1
                for iid, f1 in zip(ids_with_answers, bs["bertscore_per_item"])
            }
            for row in generation_rows:
                row["bertscore_f1"] = round(bs_per_item.get(row["id"], 0.0), 4)

            print(f"  BERTScore F1 (mean): {bs['bertscore_f1']:.4f}")

        except ImportError:
            print("  bert-score not installed — skipping BERTScore. pip install bert-score")
        except Exception as e:
            print(f"  BERTScore failed: {e}")

        gen_means = {}
        for metric in ["bleu", "rouge1", "rouge2", "rougeL", "bertscore_f1"]:
            vals = [r[metric] for r in generation_rows if metric in r]
            if vals:
                gen_means[f"{metric}_mean"] = float(np.mean(vals))

        print(f"\n  Generation Summary")
        print(f"  BLEU       : {gen_means.get('bleu_mean', 'N/A'):.4f}" if 'bleu_mean' in gen_means else "  BLEU       : N/A")
        print(f"  ROUGE-1    : {gen_means.get('rouge1_mean', 0):.4f}")
        print(f"  ROUGE-2    : {gen_means.get('rouge2_mean', 0):.4f}")
        print(f"  ROUGE-L    : {gen_means.get('rougeL_mean', 0):.4f}")
        if "bertscore_f1_mean" in gen_means:
            print(f"  BERTScore  : {gen_means['bertscore_f1_mean']:.4f}")

    # Tier 3 — RAGAS
    ragas_results = {}

    if not args.skip_ragas and not args.skip_generation and generated_answers:
        print(f"\n{'─'*65}")
        print(f"[3/3] RAGAS END-TO-END EVALUATION")
        print(f"{'─'*65}")

        ragas_rows = []
        for item in dataset:
            answer = generated_answers.get(item["id"], "")
            if not answer:
                continue
            chunks = retrieved_per_item[item["id"]]
            ragas_rows.append({
                "question"    : item["question"],
                "answer"      : answer,
                "contexts"    : [c["content"] for c in chunks],
                "ground_truth": item["ground_truth"],
            })

        print(f"  Running RAGAS on {len(ragas_rows)} items...")
        ragas_results = run_ragas(ragas_rows, model=args.openai_model)

        if ragas_results:
            print(f"\n  RAGAS Summary")
            print(f"  Faithfulness       : {ragas_results.get('faithfulness', 0):.4f}")
            print(f"  Answer Relevancy   : {ragas_results.get('answer_relevancy', 0):.4f}")
            print(f"  Context Precision  : {ragas_results.get('context_precision', 0):.4f}")
            print(f"  Context Recall     : {ragas_results.get('context_recall', 0):.4f}")

    # Save results
    print(f"\n{'─'*65}")
    print("[Saving Results]")

    save_csv(retrieval_rows, out_dir / "retrieval_results.csv")
    if generation_rows:
        save_csv(generation_rows, out_dir / "generation_results.csv")

        answers_out = [
            {
                "id"           : item["id"],
                "sector"       : item["sector"],
                "question"     : item["question"],
                "ground_truth" : item["ground_truth"],
                "generated"    : generated_answers.get(item["id"], ""),
            }
            for item in dataset
        ]
        with open(out_dir / "generated_answers.json", "w", encoding="utf-8") as f:
            json.dump(answers_out, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {out_dir / 'generated_answers.json'}")

    if ragas_results:
        ragas_export = {k: v for k, v in ragas_results.items() if k != "per_item"}
        with open(out_dir / "ragas_results.json", "w", encoding="utf-8") as f:
            json.dump(ragas_export, f, indent=2)
        print(f"  Saved: {out_dir / 'ragas_results.json'}")

    summary = {
        "evaluated_at"  : datetime.utcnow().isoformat() + "Z",
        "top_k"         : args.top_k,
        "dataset_size"  : len(dataset),
        "embed_model"   : args.embed_model,
        "llm_model"     : args.openai_model if not args.skip_generation else "skipped",
        "retrieval"     : ret_means,
    }
    if generation_rows:
        summary["generation"] = gen_means
    if ragas_results:
        summary["ragas"] = {k: v for k, v in ragas_results.items() if k != "per_item"}

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {out_dir / 'summary.json'}")

    # Generate plots
    print(f"\n{'─'*65}")
    print("[Generating Plots]")

    try:
        import matplotlib
        matplotlib.use("Agg")

        plot_retrieval_by_sector(retrieval_rows, plots_dir)
        plot_heatmap(retrieval_rows, plots_dir)
        plot_score_distribution(retrieval_rows, generation_rows, plots_dir)
        if generation_rows:
            plot_generation_by_sector(generation_rows, plots_dir)
        plot_overall_summary(summary, plots_dir)

    except ImportError:
        print("  matplotlib not installed — skipping plots. pip install matplotlib")
    except Exception as e:
        print(f"  Plotting failed: {e}")

    # Final report
    print(f"\n{'='*65}")
    print("  EVALUATION COMPLETE")
    print(f"{'='*65}")
    print(f"  Dataset     : {len(dataset)} questions")
    print(f"  Hit@{args.top_k}      : {ret_means['hit_at_k_mean']:.4f}")
    print(f"  MRR         : {ret_means['mrr_mean']:.4f}")
    print(f"  NDCG@{args.top_k}     : {ret_means['ndcg_at_k_mean']:.4f}")
    if generation_rows and "rouge1_mean" in gen_means:
        print(f"  ROUGE-1     : {gen_means['rouge1_mean']:.4f}")
        print(f"  ROUGE-L     : {gen_means['rougeL_mean']:.4f}")
    if "bertscore_f1_mean" in gen_means:
        print(f"  BERTScore   : {gen_means['bertscore_f1_mean']:.4f}")
    if ragas_results:
        print(f"  Faithfulness: {ragas_results.get('faithfulness', 0):.4f}")
        print(f"  Ans. Relev. : {ragas_results.get('answer_relevancy', 0):.4f}")
    print(f"\n  Results saved to: {out_dir.resolve()}")
    print(f"  Plots saved to  : {plots_dir.resolve()}\n")


if __name__ == "__main__":
    main()
