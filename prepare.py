"""Safely extract, audit every image, and freeze stratified duplicate-group splits."""
import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from common import ROOT, DATA, OUT, save_json, write_csv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', type=Path, default=ROOT / 'FracAtlas.zip')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip-extraction', action='store_true', help='Audit already extracted files')
    parser.add_argument('--exclude-unreadable', action='store_true', help='Record and exclude JPEG decode errors')
    parser.add_argument('--exclude-conflicting', action='store_true', help='Quarantine identical images with inconsistent labels')
    args = parser.parse_args()
    raw = DATA.parent.resolve()
    raw.mkdir(parents=True, exist_ok=True)
    if not args.skip_extraction:
        with zipfile.ZipFile(args.zip) as archive:
            for entry in archive.infolist():
                dest = (raw / entry.filename).resolve()
                if not dest.is_relative_to(raw):
                    raise ValueError(f'Unsafe ZIP path: {entry.filename}')
            archive.extractall(raw)
    with (DATA / 'dataset.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    images = defaultdict(list)
    for path in sorted((DATA / 'images').rglob('*.jpg')):
        images[path.name].append(path)
    repeated_names = {}
    for name, paths in images.items():
        if len(paths) > 1:
            hashes = {hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
            if len(hashes) != 1:
                raise ValueError(f'Different image contents share filename {name}')
            repeated_names[name] = [p.relative_to(ROOT).as_posix() for p in paths]
    if len({r['image_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate image IDs in CSV')
    groups = defaultdict(list)
    unreadable = []
    original_count = len(rows)
    for i, row in enumerate(rows):
        candidates = images[row['image_id']]
        expected = 'Fractured' if int(row['fractured']) else 'Non_fractured'
        path = next((p for p in candidates if p.parent.name == expected), candidates[0])
        with Image.open(path) as im:
            try:
                im.load()
            except OSError as exc:
                if not args.exclude_unreadable:
                    raise RuntimeError(f'Cannot decode {path}: {exc}; see --exclude-unreadable') from exc
                unreadable.append(dict(image_id=row['image_id'], fractured=int(row['fractured']), error=str(exc)))
                print(f'Excluded {path.name}: {exc}', flush=True)
                continue
            hasher = hashlib.sha256(str(im.size).encode())
            # Stream decoded RGB rows: avoid allocating several full-size pixel copies.
            for top in range(0, im.height, 64):
                with im.crop((0, top, im.width, min(top+64, im.height))) as strip:
                    with strip.convert('RGB') as rgb:
                        hasher.update(rgb.tobytes())
            digest = hasher.hexdigest()
            row['width'], row['height'] = im.size
        row['path'] = path.relative_to(ROOT).as_posix()
        row['pixel_sha256'] = digest
        groups[digest].append(i)
        if (i+1) % 500 == 0:
            print(f'Audited {i+1}/{len(rows)} images', flush=True)
    keys = sorted(groups)
    conflicts = []
    for key in list(keys):
        members = groups[key]
        if len({int(rows[i]['fractured']) for i in members}) > 1:
            conflict = dict(pixel_sha256=key, images=[dict(image_id=rows[i]['image_id'],
                            fractured=int(rows[i]['fractured'])) for i in members])
            conflicts.append(conflict)
            if not args.exclude_conflicting:
                save_json(OUT / 'conflicting_labels.json', conflicts)
                raise ValueError(f'Conflicting labels for identical images: {conflict}; see --exclude-conflicting')
            keys.remove(key)
    save_json(OUT / 'image_audit.json', dict(rows=rows, excluded_unreadable=unreadable,
                                          excluded_conflicting_groups=conflicts))
    labels = []
    for key in keys:
        values = {int(rows[i]['fractured']) for i in groups[key]}
        if len(values) != 1:
            raise ValueError(f'Conflicting labels for identical images: {key}')
        labels.append(values.pop())
    train, held = train_test_split(keys, test_size=.30, stratify=labels, random_state=args.seed)
    held_labels = [int(rows[groups[k][0]]['fractured']) for k in held]
    val, test = train_test_split(held, test_size=.5, stratify=held_labels, random_state=args.seed)
    for split, members in [('train', train), ('val', val), ('test', test)]:
        for key in members:
            for i in groups[key]:
                rows[i]['split'] = split
    rows = [r for r in rows if 'split' in r]
    write_csv(OUT / 'manifest.csv', rows)
    stats = {split: dict(Counter(int(r['fractured']) for r in rows if r['split'] == split))
             for split in ('train', 'val', 'test')}
    coco = json.loads((DATA / 'Annotations' / 'COCO JSON' / 'COCO_fracture_masks.json').read_text())
    annotated_ids = {a['image_id'] for a in coco['annotations']}
    annotated_names = {im['file_name'] for im in coco['images'] if im['id'] in annotated_ids}
    positive_names = {r['image_id'] for r in rows if int(r['fractured']) == 1}
    negative_names = {r['image_id'] for r in rows if int(r['fractured']) == 0}
    report = dict(source_csv_images=original_count, images=len(rows), excluded_unreadable=unreadable,
                  excluded_conflicting_groups=conflicts,
                  class_counts=dict(Counter(int(r['fractured']) for r in rows)),
                  splits=stats, seed=args.seed, duplicate_groups=sum(len(v)>1 for v in groups.values()),
                  duplicate_images=sum(len(v)-1 for v in groups.values()),
                  repeated_filenames_identical_content=repeated_names,
                  annotation_files=len(list((DATA / 'Annotations').rglob('*.*'))),
                  coco_regions=len(coco['annotations']),
                  positives_without_coco_regions=sorted(positive_names - annotated_names),
                  negatives_with_coco_regions=sorted(annotated_names & negative_names),
                  annotated_images_excluded_from_manifest=sorted(annotated_names - positive_names - negative_names),
                  patient_ids_available=False,
                  limitations=['Identical decoded images grouped; near duplicates and same-patient views may remain.',
                               'Subtle and previously missed fractures are not separately labeled.'])
    save_json(OUT / 'data_audit.json', report)
    print(report)

if __name__ == '__main__':
    main()
