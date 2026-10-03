"""Baseline v0: char + word TF-IDF with three logistic regression heads.

1. Fit on train, tune thresholds and temperature on validation, report metrics.
2. Refit on train + validation with the tuned settings and save artifacts/model.joblib.

Run from repo root:  python training/train_baseline.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_union

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.labels import CATEGORIES, NO_SECONDARY  # noqa: E402
from app.textprep import model_input  # noqa: E402

VERSION = "tfidf-lr-0.1"
SEED = 42


def load(name):
    d = pd.read_csv(ROOT / "data" / f"{name}.csv", keep_default_na=False, dtype=str)
    d["sec"] = d.secondary_category.replace("", NO_SECONDARY)
    d["urgent"] = d.is_urgent.str.lower() == "true"
    d["x"] = [model_input(c, s, t) for c, s, t in zip(d.channel, d.subject, d.text)]
    return d


def make_vec():
    return make_union(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=300000),
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b"),
    )


def lr():
    return LogisticRegression(C=20, max_iter=4000, class_weight="balanced", random_state=SEED)


def fit(d):
    vec = make_vec()
    X = vec.fit_transform(d.x)
    return {"vec": vec, "cat_clf": lr().fit(X, d.category), "sec_clf": lr().fit(X, d.sec), "urg_clf": lr().fit(X, d.urgent)}


def fit_temperature(p, y_idx):
    logp = np.log(np.clip(p, 1e-12, 1))

    def nll(t):
        z = logp / t
        z = z - z.max(1, keepdims=True)
        lse = np.log(np.exp(z).sum(1))
        return float(np.mean(lse - z[np.arange(len(y_idx)), y_idx]))

    return float(minimize_scalar(nll, bounds=(0.2, 5.0), method="bounded").x)


def ece(conf, correct, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def main():
    from app.model import Predictor  # reuse the exact serving postprocess

    tr, va = load("train"), load("validation")
    allowed = sorted({(c, s) for c, s in zip(tr.category, tr.sec) if s != NO_SECONDARY}
                     | {(c, s) for c, s in zip(va.category, va.sec) if s != NO_SECONDARY})

    # ---- stage 1: fit on train, tune on validation
    m = fit(tr)
    Xva = m["vec"].transform(va.x)
    p_cat = m["cat_clf"].predict_proba(Xva)
    cat_classes = list(m["cat_clf"].classes_)
    y_idx = np.array([cat_classes.index(c) for c in va.category])
    temperature = fit_temperature(p_cat, y_idx)

    base = {"kind": "tfidf_lr", "version": VERSION, "cat_classes": cat_classes,
            "sec_classes": list(m["sec_clf"].classes_), "allowed_pairs": [list(p) for p in allowed],
            "review_threshold": 0.5, "temperature": temperature, **m}

    records = va[["channel", "subject", "text"]].to_dict("records")
    cached = (p_cat, m["sec_clf"].predict_proba(Xva),
              m["urg_clf"].predict_proba(Xva)[:, list(m["urg_clf"].classes_).index(True)])
    best = None
    for st in np.arange(0.20, 0.81, 0.05):
        for ut in np.arange(0.20, 0.81, 0.05):
            pr = Predictor(art={**base, "sec_threshold": float(st), "urgent_threshold": float(ut)})
            pr.backend.probs = lambda texts, _c=cached: _c  # reuse validation probabilities
            preds = pr.predict(records)
            sec = [p["secondary_category"] or NO_SECONDARY for p in preds]
            urg = [p["is_urgent"] for p in preds]
            score = f1_score(va.sec, sec, average="macro") + f1_score(va.urgent, urg, average="macro")
            if best is None or score > best[0]:
                best = (score, float(st), float(ut), preds)
    _, sec_t, urg_t, preds = best

    cat_pred = np.array([p["category"] for p in preds])
    conf = np.array([p["confidence"] for p in preds])
    correct = (cat_pred == va.category.values).astype(float)
    va["pred"] = cat_pred
    metrics = {
        "model_version_tag": VERSION,
        "validation": {
            "category_macro_f1": round(f1_score(va.category, cat_pred, average="macro"), 4),
            "category_accuracy": round(float(correct.mean()), 4),
            "secondary_macro_f1": round(f1_score(va.sec, [p["secondary_category"] or NO_SECONDARY for p in preds], average="macro"), 4),
            "urgent_f1": round(f1_score(va.urgent, [p["is_urgent"] for p in preds]), 4),
            "category_ece": round(ece(conf, correct), 4),
            "accuracy_by_language": va.groupby("language").apply(lambda g: round(float((g.pred == g.category).mean()), 4)).to_dict(),
        },
        "tuned": {"sec_threshold": sec_t, "urgent_threshold": urg_t, "temperature": round(temperature, 4)},
    }
    print(json.dumps(metrics, indent=2))

    # ---- stage 2: refit on train + validation and save
    full = pd.concat([tr, va], ignore_index=True)
    mf = fit(full)
    art = {**base, **mf, "sec_threshold": sec_t, "urgent_threshold": urg_t,
           "cat_classes": list(mf["cat_clf"].classes_), "sec_classes": list(mf["sec_clf"].classes_)}
    assert art["cat_classes"] == sorted(CATEGORIES)
    out = ROOT / "artifacts" / "model.joblib"
    joblib.dump(art, out, compress=3)
    (ROOT / "artifacts" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("saved", out, "->", Predictor(out).version)


if __name__ == "__main__":
    main()
