"""Score a local X-ray using a trusted project checkpoint."""
import argparse
import json
from pathlib import Path
from common import OUT, image_rgb

def main():
    p = argparse.ArgumentParser()
    p.add_argument('image', type=Path)
    p.add_argument('--model', choices=['baseline', 'cnn'], default='baseline')
    args = p.parse_args()
    if args.model == 'baseline':
        import joblib
        from train_baseline import features
        bundle = joblib.load(OUT / 'baseline' / 'model.joblib')
        probability = float(bundle['model'].predict_proba(features(args.image)[None])[0, 1])
    else:
        import torch
        from torchvision import models, transforms
        bundle = torch.load(OUT / 'cnn' / 'model.pt', map_location='cpu', weights_only=True)
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 1)
        model.load_state_dict(bundle['state_dict'])
        model.eval()
        transform = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        with torch.no_grad():
            probability = float(model(transform(image_rgb(args.image, bundle['size']))[None]).sigmoid().item())
    print(json.dumps(dict(image=str(args.image), fracture_score=probability,
        threshold=bundle['threshold'], fracture_flag=probability >= bundle['threshold'],
        note='Research score, not a calibrated clinical probability or diagnosis.'), indent=2))

if __name__ == '__main__':
    main()
