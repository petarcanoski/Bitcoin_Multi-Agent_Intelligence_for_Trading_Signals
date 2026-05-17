
import sys
import time
import numpy as np
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "sentiment_analysis"))


def _load_phrasebank():
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install with: pip install datasets")
    ds = load_dataset("financial_phrasebank", "sentences_75agree",
                      trust_remote_code=True)
    return [(row["sentence"], int(row["label"])) for row in ds["train"]]


def _load_finbert():
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError:
        raise ImportError("Install with: pip install transformers torch")

    model_id = "ProsusAI/finbert"
    print(f"Loading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()
    return tokenizer, model


_FINBERT_TO_PB = {0: 2, 1: 0, 2: 1}


def _predict_batch(tokenizer, model, texts: list, batch_size: int = 32):
    import torch
    all_probs, all_preds = [], []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i: i + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", truncation=True,
                        max_length=512, padding=True)
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=-1)
        all_probs.append(probs.numpy())
        all_preds.extend(probs.argmax(dim=-1).tolist())
    return np.vstack(all_probs), all_preds


def evaluate() -> dict:
    print("=" * 60)
    print("SENTIMENT AGENT EVALUATION")
    print("=" * 60)

    print("\nLoading financial_phrasebank (sentences_75agree)...")
    try:
        samples = _load_phrasebank()
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        return {"error": str(e)}
    print(f"Loaded {len(samples)} samples")

    label_names = {0: "negative", 1: "neutral", 2: "positive"}
    true_labels = [lbl for _, lbl in samples]
    true_dist = Counter(true_labels)
    print("Label distribution:")
    for lbl, name in label_names.items():
        print(f"  {name:>10}: {true_dist[lbl]:4d} ({true_dist[lbl]/len(samples):.1%})")

    try:
        tokenizer, model = _load_finbert()
    except Exception as e:
        print(f"ERROR loading FinBERT: {e}")
        return {"error": str(e)}

    texts = [s for s, _ in samples]
    print(f"\nRunning FinBERT inference on {len(texts)} samples...")
    t0 = time.time()
    all_probs, finbert_preds = _predict_batch(tokenizer, model, texts)
    elapsed = time.time() - t0
    print(f"Inference done in {elapsed:.1f}s")

    pred_labels = [_FINBERT_TO_PB[p] for p in finbert_preds]

    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix
    )
    true_arr = np.array(true_labels)
    pred_arr = np.array(pred_labels)

    acc = accuracy_score(true_arr, pred_arr)
    cm = confusion_matrix(true_arr, pred_arr, labels=[0, 1, 2])
    report_str = classification_report(
        true_arr, pred_arr, labels=[0, 1, 2],
        target_names=["negative", "neutral", "positive"], digits=4,
    )
    report_dict = classification_report(
        true_arr, pred_arr, labels=[0, 1, 2],
        target_names=["negative", "neutral", "positive"], digits=4,
        output_dict=True,
    )

    print(f"\nAccuracy : {acc:.4f} ({acc:.2%})")
    print(report_str)
    print(f"Confusion matrix (rows=true, cols=pred):")
    print(f"  {'':12} neg   neu   pos")
    for i, name in enumerate(["negative", "neutral", "positive"]):
        row = "  ".join(f"{cm[i, j]:5d}" for j in range(3))
        print(f"  {name:>12} {row}")

    sentiment_scores = all_probs[:, 0] - all_probs[:, 1]
    avg_by_true_class = {
        label_names[lbl]: float(sentiment_scores[true_arr == lbl].mean())
        for lbl in [0, 1, 2]
    }
    print(f"\nAvg sentiment score by true class (positive - negative):")
    for cls, avg in avg_by_true_class.items():
        print(f"  {cls:>10}: {avg:+.4f}")

    wrong_mask = pred_arr != true_arr
    n_wrong = int(wrong_mask.sum())
    print(f"\nMisclassifications: {n_wrong} / {len(samples)} ({n_wrong/len(samples):.1%})")
    wrong_pairs = Counter(
        (label_names[true_arr[i]], label_names[pred_arr[i]])
        for i in range(len(true_arr)) if wrong_mask[i]
    )
    print("Most common mistakes (true → predicted):")
    for (true_cls, pred_cls), cnt in wrong_pairs.most_common(5):
        print(f"  {true_cls:>10} → {pred_cls:<10}: {cnt}")

    return {
        "dataset": "financial_phrasebank (sentences_75agree)",
        "n_samples": len(samples),
        "model": "ProsusAI/finbert",
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "report": report_dict,
        "avg_sentiment_score_by_class": avg_by_true_class,
        "inference_seconds": elapsed,
        "label_distribution": {
            label_names[k]: int(v) for k, v in true_dist.items()
        },
        "misclassification_count": n_wrong,
    }


if __name__ == "__main__":
    evaluate()
