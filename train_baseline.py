"""CPU baseline using whole-image intensity and gradient features. Not a clinical model."""
import argparse
import hashlib
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import ROOT, OUT, image_rgb, read_manifest, evaluate, save_json

def features(path):
    a = np.asarray(image_rgb(path, 64).convert('L'), dtype=np.float32) / 255
    gy, gx = np.gradient(a)
    # Spatial gradients retain a coarse representation of bone boundaries.
    return np.concatenate([a.ravel(), gx.ravel(), gy.ravel()])

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--target-sensitivity', type=float, default=.95)
    p.add_argument('--c', type=float, default=.01)
    args = p.parse_args()
    rows = read_manifest()
    run = OUT / 'baseline'
    run.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256((OUT / 'manifest.csv').read_bytes()).hexdigest()
    cache = OUT / 'baseline_features.npz'
    if cache.exists():
        with np.load(cache) as saved:
            x = saved['x'] if str(saved['fingerprint']) == fingerprint else None
    else:
        x = None
    if x is None:
        x = np.empty((len(rows), 64*64*3), dtype=np.float32)
        for i, r in enumerate(rows):
            x[i] = features(ROOT / r['path'])
            if (i+1) % 500 == 0:
                print(f'Processed {i+1}/{len(rows)} images', flush=True)
        np.savez_compressed(cache, x=x, fingerprint=fingerprint)
    y = np.array([int(r['fractured']) for r in rows])
    split = np.array([r['split'] for r in rows])
    model = make_pipeline(StandardScaler(), LogisticRegression(C=args.c, class_weight='balanced',
                                                               max_iter=2000, random_state=42))
    model.fit(x[split == 'train'], y[split == 'train'])
    threshold = evaluate(run, rows, model.predict_proba(x[split == 'val'])[:, 1],
                         model.predict_proba(x[split == 'test'])[:, 1], args.target_sensitivity)
    joblib.dump({'model': model, 'threshold': threshold, 'feature_version': 1}, run / 'model.joblib')
    save_json(run / 'config.json', dict(vars(args), manifest_sha256=fingerprint,
                                      model='balanced logistic regression on 64px image/gradient features'))

if __name__ == '__main__':
    main()
