"""Model wrapper. Every backend returns raw probabilities; postprocess() applies the contract rules.

Artifact format (joblib dict):
  kind              "tfidf_lr" (baseline) or "onnx_encoder" (added later)
  version           human version tag, e.g. "tfidf-lr-0.1"
  cat_classes       list of categories in probability column order
  sec_classes       list of secondary labels incl. "none"
  allowed_pairs     list of [primary, secondary] pairs seen in training
  sec_threshold     min prob for a secondary label
  urgent_threshold  min prob for is_urgent
  review_threshold  confidence below this sets needs_human_review
  temperature       softmax temperature for the category head (calibration)
"""
import hashlib
from pathlib import Path

import joblib
import numpy as np

from .labels import NO_SECONDARY, TEAM_BY_CATEGORY
from .textprep import model_input


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:8]


def _softmax_t(logp: np.ndarray, t: float) -> np.ndarray:
    z = logp / t
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class TfidfBackend:
    def __init__(self, art):
        self.vec = art["vec"]
        self.cat_clf = art["cat_clf"]
        self.sec_clf = art["sec_clf"]
        self.urg_clf = art["urg_clf"]

    def probs(self, texts):
        X = self.vec.transform(texts)
        p_cat = self.cat_clf.predict_proba(X)
        p_sec = self.sec_clf.predict_proba(X)
        p_urg = self.urg_clf.predict_proba(X)[:, list(self.urg_clf.classes_).index(True)]
        return p_cat, p_sec, p_urg


BACKENDS = {"tfidf_lr": TfidfBackend}


class Predictor:
    def __init__(self, path: Path | None = None, art: dict | None = None):
        if art is None:
            art = joblib.load(path)
        self.backend = BACKENDS[art["kind"]](art)
        self.cat_classes = list(art["cat_classes"])
        self.sec_classes = list(art["sec_classes"])
        self.allowed_pairs = {tuple(p) for p in art["allowed_pairs"]}
        self.sec_threshold = float(art["sec_threshold"])
        self.urgent_threshold = float(art["urgent_threshold"])
        self.review_threshold = float(art["review_threshold"])
        self.temperature = float(art.get("temperature", 1.0))
        self.version = f'{art["version"]}+{_file_hash(path)}' if path else art["version"]

    def predict(self, tickets):
        """tickets: list of clean dicts with channel, subject, text and optional ticket_id."""
        if not tickets:
            return []
        texts = [model_input(t["channel"], t.get("subject", ""), t["text"]) for t in tickets]
        p_cat, p_sec, p_urg = self.backend.probs(texts)
        if self.temperature != 1.0:
            p_cat = _softmax_t(np.log(np.clip(p_cat, 1e-12, 1)), self.temperature)
        out = []
        for i, t in enumerate(tickets):
            pred = self.postprocess(p_cat[i], p_sec[i], float(p_urg[i]))
            if "ticket_id" in t:
                pred = {"ticket_id": t["ticket_id"], **pred}
            out.append(pred)
        return out

    def postprocess(self, p_cat, p_sec, p_urg):
        k = int(np.argmax(p_cat))
        category = self.cat_classes[k]
        confidence = float(np.clip(p_cat[k], 0.0, 1.0))

        secondary = None
        best, best_p = None, -1.0
        for j, s in enumerate(self.sec_classes):
            if s in (NO_SECONDARY, category) or (category, s) not in self.allowed_pairs:
                continue
            if p_sec[j] > best_p:
                best, best_p = s, float(p_sec[j])
        if best is not None and best_p >= self.sec_threshold:
            secondary = best

        is_urgent = p_urg >= self.urgent_threshold
        if category == "spam_irrelevant":
            secondary, is_urgent = None, False

        return {
            "category": category,
            "secondary_category": secondary,
            "team": TEAM_BY_CATEGORY[category],
            "is_urgent": bool(is_urgent),
            "confidence": round(confidence, 4),
            "model_version": self.version,
            "needs_human_review": confidence < self.review_threshold,
        }
