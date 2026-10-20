"""Make a local error-review sheet with reference bounding boxes and mask polygons."""
import argparse
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from PIL import Image
from common import ROOT, DATA, OUT

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['baseline', 'cnn'], default='baseline')
    p.add_argument('--limit', type=int, default=12)
    args = p.parse_args()
    run = OUT / args.model
    with (run / 'test_predictions.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    rows = sorted([r for r in rows if r['false_negative'] == '1'], key=lambda r: float(r['probability']))[:args.limit]
    if not rows:
        print('No false negatives at the selected threshold.')
        return
    coco = json.loads((DATA / 'Annotations' / 'COCO JSON' / 'COCO_fracture_masks.json').read_text())
    ids = {PathName(r['file_name']): r['id'] for r in coco['images']}
    fig, axes = plt.subplots((len(rows)+2)//3, 3, figsize=(15, 5*((len(rows)+2)//3)), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for ax, row in zip(axes.flat, rows):
        with Image.open(ROOT / row['path']) as im:
            width, height = im.size
            im.thumbnail((768, 768))
            ax.imshow(im.convert('RGB'), extent=(0, width, height, 0))
        for ann in coco['annotations']:
            if ann['image_id'] != ids.get(row['image_id']):
                continue
            x, y, w, h = ann['bbox']
            ax.add_patch(Rectangle((x,y), w,h, fill=False, edgecolor='lime', linewidth=1.5))
            for poly in ann.get('segmentation', []) if isinstance(ann.get('segmentation'), list) else []:
                ax.add_patch(Polygon(list(zip(poly[::2], poly[1::2])), fill=False, edgecolor='cyan'))
        ax.set_title(f"{row['image_id']} | score {float(row['probability']):.2g}")
    fig.suptitle('MISSED FRACTURES — green boxes / cyan outlines are HUMAN reference annotations', fontsize=14)
    fig.tight_layout(rect=(0,0,1,.97))
    fig.savefig(run / 'missed_fractures.png', dpi=140)
    plt.close(fig)
    print(run / 'missed_fractures.png')

def PathName(value):
    return value.replace('\\', '/').split('/')[-1]

if __name__ == '__main__':
    main()
