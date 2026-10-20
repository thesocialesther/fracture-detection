"""Visualize the audited data and replay the saved classifier on every test image."""
import csv
import hashlib
import html
import json
import os
import time
from urllib.parse import quote
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from common import ROOT, OUT, read_manifest, image_rgb, metrics, write_csv, save_json
from train_baseline import features

DEST = OUT / 'test_replay'
COLORS = ['#2875a7', '#de7747']

def savefig(fig, name):
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(DEST / name, dpi=140, bbox_inches='tight')
    plt.close(fig)

def main():
    start = time.perf_counter()
    DEST.mkdir(parents=True, exist_ok=True)
    rows = read_manifest()
    audit = json.loads((OUT / 'data_audit.json').read_text())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    counts = [sum(int(r['fractured']) == label for r in rows) for label in (0, 1)]
    bars = axes[0, 0].bar(['No fracture', 'Fracture'], counts, color=COLORS)
    axes[0, 0].bar_label(bars, padding=3)
    axes[0, 0].set_title('Usable images: class imbalance')
    axes[0, 0].set_ylim(0, max(counts)*1.15)
    parts = ['train', 'val', 'test']
    normal = [sum(r['split'] == s and r['fractured'] == '0' for r in rows) for s in parts]
    fracture = [sum(r['split'] == s and r['fractured'] == '1' for r in rows) for s in parts]
    a = axes[0, 1]
    a.bar(parts, normal, color=COLORS[0], label='No fracture')
    a.bar(parts, fracture, bottom=normal, color=COLORS[1], label='Fracture')
    for i, (n, f) in enumerate(zip(normal, fracture)):
        a.text(i, n+f+45, str(n+f), ha='center')
    a.set_ylim(0, max(np.array(normal)+fracture)*1.2)
    a.set_title('Frozen split sizes')
    a.legend()
    anatomy = ['hand', 'leg', 'hip', 'shoulder']
    x = np.arange(len(anatomy))
    a = axes[1, 0]
    a.bar(x-.18, [sum(r[k]=='1' and r['fractured']=='0' for r in rows) for k in anatomy],
          width=.36, color=COLORS[0], label='No fracture')
    a.bar(x+.18, [sum(r[k]=='1' and r['fractured']=='1' for r in rows) for k in anatomy],
          width=.36, color=COLORS[1], label='Fracture')
    a.set_xticks(x, anatomy)
    a.set_title('Anatomy flags (may overlap)')
    a.legend()
    a = axes[1, 1]
    a.axis('off')
    conflicting = sum(len(g['images']) for g in audit['excluded_conflicting_groups'])
    a.text(.03, .9, f"Source CSV: {audit['source_csv_images']:,} images\n\n"
        f"Usable: {len(rows):,} images\n\n"
        f"Excluded: {len(audit['excluded_unreadable'])} unreadable + {conflicting} conflicting labels\n\n"
        'Exact duplicates stay in a single split.\n\n'
        'Patient independence is unverified.\n\n'
        'Subtle / previously missed labels are absent.', va='top', fontsize=12)
    fig.suptitle('FracAtlas: audited dataset overview', fontsize=17)
    savefig(fig, 'dataset_overview.png')
    print('Dataset visualization saved.', flush=True)

    checkpoint = OUT / 'baseline' / 'model.joblib'
    model_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    bundle = joblib.load(checkpoint)
    threshold = float(bundle['threshold'])
    test = [r for r in rows if r['split'] == 'test']
    # Reopen and preprocess every original test image; do not reuse cached features or predictions.
    probabilities = []
    for offset in range(0, len(test), 32):
        batch = test[offset:offset+32]
        x = np.stack([features(ROOT / r['path']) for r in batch])
        probabilities.extend(bundle['model'].predict_proba(x)[:, 1].tolist())
        print(f'Predicted {min(offset+32, len(test))}/{len(test)} test images', flush=True)
    y = np.array([int(r['fractured']) for r in test])
    probabilities = np.asarray(probabilities)
    predicted = probabilities >= threshold
    results = []
    for r, prob, pred in zip(test, probabilities, predicted):
        outcome = ('TP' if pred else 'FN') if int(r['fractured']) else ('FP' if pred else 'TN')
        results.append(dict(r, fracture_score=float(prob), predicted_label=int(pred),
                            threshold=threshold, outcome=outcome))
    write_csv(DEST / 'test_predictions.csv', results)
    report = metrics(y, probabilities, threshold)
    old = {r['image_id']: float(r['probability']) for r in csv.DictReader(
        (OUT / 'baseline' / 'test_predictions.csv').open(newline=''))}
    difference = max(abs(r['fracture_score']-old[r['image_id']]) for r in results)
    assert difference < 1e-6, 'Fresh inference differs from original saved predictions'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == model_hash
    save_json(DEST / 'inference_summary.json', dict(metrics=report,
        checkpoint_sha256=model_hash, manifest_sha256=hashlib.sha256((OUT/'manifest.csv').read_bytes()).hexdigest(),
        model_retrained=False, threshold_retuned=False, fresh_images_scored=len(test),
        max_difference_from_original_predictions=difference, elapsed_seconds=time.perf_counter()-start))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ConfusionMatrixDisplay.from_predictions(y, predicted, display_labels=['No fracture', 'Fracture'],
        ax=axes[0], cmap='Blues', colorbar=False)
    axes[0].set_title(f"Test confusion matrix\n{report['fn']} missed fractures, {report['fp']} false alarms")
    RocCurveDisplay.from_predictions(y, probabilities, ax=axes[1])
    axes[1].plot([0,1], [0,1], '--', color='gray')
    axes[1].set_title('ROC curve')
    PrecisionRecallDisplay.from_predictions(y, probabilities, ax=axes[2])
    axes[2].axhline(y.mean(), linestyle='--', color='gray', label='Fracture prevalence')
    axes[2].legend(fontsize=8)
    axes[2].set_title('Precision-recall curve')
    fig.suptitle(f"Saved baseline: {len(test)} fresh test predictions | sensitivity {report['sensitivity']:.1%} | specificity {report['specificity']:.1%}", fontsize=14)
    savefig(fig, 'test_results.png')

    fig, axes = plt.subplots(4, 3, figsize=(12, 14))
    titles = {'TP': 'Correct: fracture', 'FN': 'MISSED fracture',
              'FP': 'False alarm: no fracture', 'TN': 'Correct: no fracture'}
    for line, outcome in enumerate(['TP', 'FN', 'FP', 'TN']):
        examples = sorted([r for r in results if r['outcome']==outcome], key=lambda r:r['image_id'])[:3]
        for ax in axes[line]:
            ax.axis('off')
        for ax, r in zip(axes[line], examples):
            ax.imshow(image_rgb(ROOT/r['path'], 512))
            ax.set_title(f"{titles[outcome]}\n{r['image_id']} | score {r['fracture_score']:.2g}",
                         color='#b22b2b' if outcome in ('FN', 'FP') else '#17633e', fontsize=10)
    fig.suptitle('Test examples: first 3 image IDs in each outcome group\nImage-level predictions; scores are not calibrated probabilities', fontsize=14)
    savefig(fig, 'prediction_examples.png')

    cards = []
    for r in sorted(results, key=lambda r: (['FN','FP','TP','TN'].index(r['outcome']), r['image_id'])):
        src = quote(os.path.relpath(ROOT/r['path'], DEST).replace('\\', '/'), safe='/')
        cards.append(f'<article><h3>{html.escape(titles[r["outcome"]])}</h3>'
            f'<img loading="lazy" src="{src}" alt="X-ray {html.escape(r["image_id"])}">'
            f'<p>{html.escape(r["image_id"])}<br>Reference: {"fracture" if r["fractured"]=="1" else "no fracture"}'
            f'<br>Prediction: {"fracture" if r["predicted_label"] else "no fracture"}'
            f'<br>Score: {r["fracture_score"]:.5g}</p></article>')
    page = '<!doctype html><html><meta charset="utf-8"><title>FracAtlas test review</title>'
    page += '<style>body{font:16px system-ui;margin:30px;background:#f4f6f8;color:#172633}section{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}article{background:white;padding:16px;border-radius:10px}article img{width:100%;height:300px;object-fit:contain;background:black}.chart{width:100%;max-width:1400px}</style>'
    page += f'<h1>FracAtlas: data and saved-model test review</h1><p>{len(test)} fresh predictions. Threshold {threshold:.6g}, fixed from the original validation run. No retraining or threshold adjustment.</p>'
    page += f'<p>Sensitivity {report["sensitivity"]:.1%}; specificity {report["specificity"]:.1%}; missed fractures {report["fn"]}/{report["tp"]+report["fn"]}; false alarms {report["fp"]}.</p>'
    page += '<p>Research baseline. Scores are uncalibrated. No predicted injury locations. Patient independence is unverified.</p>'
    for file in ['dataset_overview.png', 'test_results.png', 'prediction_examples.png']:
        page += f'<img class="chart" src="{file}" alt="{file}">'
    page += '<h2>All test images</h2><p>Ordered: missed fractures, false alarms, correctly flagged fractures, correctly rejected normal images.</p><section>' + ''.join(cards) + '</section></html>'
    (DEST/'index.html').write_text(page, encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    print(f'Open {DEST / "index.html"}', flush=True)

if __name__ == '__main__':
    main()
