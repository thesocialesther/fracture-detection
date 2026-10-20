"""ResNet18 transfer learning, weighted loss, validation-only selection and threshold."""
import argparse
import copy
import hashlib
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import average_precision_score
from common import ROOT, OUT, image_rgb, read_manifest, evaluate, save_json

class Radiographs(Dataset):
    def __init__(self, rows, size, augment=False):
        self.rows, self.size = rows, size
        ops = [transforms.RandomRotation(7)] if augment else []
        self.transform = transforms.Compose(ops + [transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return self.transform(image_rgb(ROOT / r['path'], self.size)), torch.tensor(float(r['fractured']))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--size', type=int, default=384)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--target-sensitivity', type=float, default=.95)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--from-scratch', action='store_true', help='Skip ImageNet weights download')
    args = p.parse_args()
    if args.epochs < 1 or args.size < 32:
        p.error('Use positive epochs and size >=32')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows = read_manifest()
    parts = {s: [r for r in rows if r['split'] == s] for s in ('train', 'val', 'test')}
    loaders = {s: DataLoader(Radiographs(v, args.size, s == 'train'), batch_size=args.batch_size,
                            shuffle=s == 'train', num_workers=0) for s, v in parts.items()}
    # Keep downloaded pretrained weights inside this project.
    torch.hub.set_dir(str(OUT / 'torch_cache'))
    model = models.resnet18(weights=None if args.from_scratch else models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.to(device)
    npos = sum(int(r['fractured']) for r in parts['train'])
    weight = (len(parts['train']) - npos) / npos
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    run = OUT / 'cnn'
    run.mkdir(parents=True, exist_ok=True)
    best, stale, history, state = -1., 0, [], None

    def predict(split):
        model.eval()
        values = []
        with torch.no_grad():
            for x, _ in loaders[split]:
                values.extend(model(x.to(device)).flatten().sigmoid().cpu().tolist())
        return np.asarray(values)

    for epoch in range(args.epochs):
        model.train()
        total = 0.
        for x, y in loaders['train']:
            optimizer.zero_grad()
            loss = loss_fn(model(x.to(device)).flatten(), y.to(device))
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
        vp = predict('val')
        ap = float(average_precision_score([int(r['fractured']) for r in parts['val']], vp))
        history.append(dict(epoch=epoch+1, train_loss=total/len(parts['train']), val_average_precision=ap))
        print(history[-1], flush=True)
        if ap > best:
            best, stale = ap, 0
            state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
        save_json(run / 'history.json', history)
        if stale >= args.patience:
            break
    model.load_state_dict(state)
    # Test set is first accessed after all epoch/model selection is complete.
    threshold = evaluate(run, rows, predict('val'), predict('test'), args.target_sensitivity)
    torch.save(dict(state_dict=state, threshold=threshold, size=args.size), run / 'model.pt')
    save_json(run / 'config.json', dict(vars(args), device=str(device), positive_weight=weight,
        manifest_sha256=hashlib.sha256((OUT / 'manifest.csv').read_bytes()).hexdigest()))

if __name__ == '__main__':
    main()
