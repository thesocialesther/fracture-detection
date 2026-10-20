"""Shared paths, preprocessing, and evaluation; all paths are project-relative."""
from pathlib import Path
import csv
import json
import numpy as np
from PIL import Image, ImageOps
from sklearn.metrics import (confusion_matrix, roc_auc_score, average_precision_score,
                             roc_curve, RocCurveDisplay, PrecisionRecallDisplay)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'raw' / 'FracAtlas'
OUT = ROOT / 'artifacts'

def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')

def read_manifest():
    with (OUT / 'manifest.csv').open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def image_rgb(path, size):
    # Preserve the entire radiograph: no center crop that could remove a fracture.
    with Image.open(path) as im:
        im.thumbnail((size, size), Image.Resampling.LANCZOS)
        return ImageOps.pad(im.convert('RGB'), (size, size), color=(0, 0, 0))

def threshold_for_sensitivity(y, p, target):
    if not 0 < target <= 1:
        raise ValueError('Target sensitivity must be in (0, 1].')
    fpr, tpr, thresholds = roc_curve(y, p, drop_intermediate=False)
    valid = np.flatnonzero((tpr >= target) & np.isfinite(thresholds))
    # Highest threshold satisfying target; avoids unnecessarily flagging more negatives.
    return float(np.max(thresholds[valid]))

def metrics(y, p, threshold):
    y, p = np.asarray(y, int), np.asarray(p, float)
    tn, fp, fn, tp = confusion_matrix(y, p >= threshold, labels=[0, 1]).ravel()
    def ratio(a, b):
        return float(a / b) if b else None
    sensitivity = ratio(tp, tp + fn)
    # Wilson interval describes sampling uncertainty, not clinical validity.
    n = int(tp + fn)
    z = 1.96
    center = (sensitivity + z*z/(2*n)) / (1 + z*z/n)
    half = z * np.sqrt(sensitivity*(1-sensitivity)/n + z*z/(4*n*n)) / (1+z*z/n)
    return dict(threshold=float(threshold), n=len(y), fracture_prevalence=float(y.mean()), tp=int(tp), tn=int(tn),
                fp=int(fp), fn=int(fn), sensitivity=sensitivity,
                sensitivity_ci95_wilson=[float(center-half), float(center+half)],
                specificity=ratio(tn, tn+fp), precision=ratio(tp, tp+fp),
                negative_predictive_value=ratio(tn, tn+fn),
                false_negative_rate=ratio(fn, tp+fn), accuracy=ratio(tp+tn, len(y)),
                roc_auc=float(roc_auc_score(y, p)),
                average_precision=float(average_precision_score(y, p)))

def evaluate(run, rows, val_prob, test_prob, target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    val = [r for r in rows if r['split'] == 'val']
    test = [r for r in rows if r['split'] == 'test']
    vy = np.array([int(r['fractured']) for r in val])
    ty = np.array([int(r['fractured']) for r in test])
    threshold = threshold_for_sensitivity(vy, val_prob, target)
    report = {'target_validation_sensitivity': target,
              'validation': metrics(vy, val_prob, threshold),
              'test': metrics(ty, test_prob, threshold),
              'test_at_0.5': metrics(ty, test_prob, .5),
              'threshold_selected_on': 'validation only',
              'limitation': 'Image-level research split; patient independence unverified. No subtlety labels.'}
    save_json(run / 'metrics.json', report)
    predictions = [dict(r, probability=float(p), prediction=int(p >= threshold),
                        false_negative=int(int(r['fractured']) == 1 and p < threshold))
                   for r, p in zip(test, test_prob)]
    write_csv(run / 'test_predictions.csv', predictions)
    missed = [r for r in predictions if r['false_negative']]
    # Keep a header even when no fractures were missed.
    write_csv(run / 'false_negatives.csv', missed, fieldnames=list(predictions[0]))
    for field in ('hand', 'leg', 'hip', 'shoulder', 'hardware', 'multiscan'):
        ix = [i for i, r in enumerate(test) if r[field] == '1']
        if ix and len(set(ty[ix])) == 2:
            report.setdefault('test_subgroups', {})[field] = metrics(ty[ix], np.asarray(test_prob)[ix], threshold)
    save_json(run / 'metrics.json', report)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    RocCurveDisplay.from_predictions(ty, test_prob, ax=axes[0])
    PrecisionRecallDisplay.from_predictions(ty, test_prob, ax=axes[1])
    fig.tight_layout()
    fig.savefig(run / 'curves.png', dpi=150)
    plt.close(fig)
    print(json.dumps(report, indent=2))
    return threshold
