"""Package source, checkpoint and results for GitHub without raw data or caches."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parent

def selected_files():
    excluded_dirs = {'data', '.git', '.venv', '.packages', '__pycache__', '.github-upload', 'torch_cache'}
    for path in sorted(ROOT.rglob('*')):
        relative = path.relative_to(ROOT)
        if not path.is_file() or any(part in excluded_dirs for part in relative.parts):
            continue
        if path.suffix in {'.zip', '.npz', '.pyc', '.pyo'} or path.name.startswith('.env'):
            continue
        if path.name in {'UPLOAD_MANIFEST.json', 'Thumbs.db', '.DS_Store'}:
            continue
        yield path

def main():
    files = list(selected_files())
    entries = [dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size,
                    sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
    manifest = dict(files=entries, excluded=['data/', 'FracAtlas.zip', 'artifacts/baseline_features.npz',
        'artifacts/torch_cache/', '.venv/', '__pycache__/', 'upload ZIPs and environment secrets'],
        dataset_source='https://doi.org/10.6084/m9.figshare.22363012')
    text = json.dumps(manifest, indent=2)
    (ROOT/'UPLOAD_MANIFEST.json').write_text(text, encoding='utf-8')
    archive = ROOT/'fracture-detection-github.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())
        z.writestr('UPLOAD_MANIFEST.json', text)
    print(f'Packaged {len(entries)+1} files; {archive.stat().st_size:,} bytes: {archive}')

if __name__ == '__main__':
    main()
