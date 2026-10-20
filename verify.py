"""Check evaluation and frozen split integrity."""
import numpy as np
from collections import defaultdict
from common import ROOT, read_manifest, threshold_for_sensitivity, metrics

def main():
    y = np.array([0, 0, 1, 1, 1])
    p = np.array([.1, .8, .2, .7, .9])
    m = metrics(y, p, .5)
    assert (m['tn'], m['fp'], m['fn'], m['tp']) == (1, 1, 1, 2)
    t = threshold_for_sensitivity(y, p, 1.)
    assert t == .2 and metrics(y, p, t)['fn'] == 0
    assert metrics(y, p, np.nextafter(t, np.inf))['fn'] == 1
    assert threshold_for_sensitivity([0, 1, 1], [.8, .4, .4], .95) == .4
    rows = read_manifest()
    groups, labels, ids = defaultdict(set), defaultdict(set), set()
    for r in rows:
        assert r['image_id'] not in ids
        ids.add(r['image_id'])
        assert (ROOT / r['path']).is_file()
        groups[r['pixel_sha256']].add(r['split'])
        labels[r['split']].add(int(r['fractured']))
    assert all(len(v) == 1 for v in groups.values()), 'Duplicate leakage'
    assert set(labels) == {'train', 'val', 'test'}
    assert all(v == {0, 1} for v in labels.values())
    print(f'Passed evaluation and split integrity checks on {len(rows)} images.')

if __name__ == '__main__':
    main()
