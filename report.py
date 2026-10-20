"""Write a readable summary of an actual completed run."""
import argparse
import json
from common import ROOT, OUT

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['baseline', 'cnn'], default='baseline')
    args = p.parse_args()
    audit = json.loads((OUT / 'data_audit.json').read_text())
    report = json.loads((OUT / args.model / 'metrics.json').read_text())
    v, t, default = report['validation'], report['test'], report['test_at_0.5']
    def pct(value):
        return 'undefined' if value is None else f'{100*value:.1f}%'
    lines = [f'# FracAtlas results: {args.model}', '',
        'This report summarizes an actual completed image-level research experiment.', '',
        f"Source images: {audit['source_csv_images']}. Usable images: {audit['images']}. "
        f"Excluded unreadable images: {len(audit['excluded_unreadable'])}. "
        f"Quarantined conflicting groups: {len(audit['excluded_conflicting_groups'])}.", '',
        f"Class counts (0=no fracture, 1=fracture): {audit['class_counts']}.", '',
        f"Frozen splits: {audit['splits']}.", '',
        f"Target validation sensitivity: {pct(report['target_validation_sensitivity'])}. "
        f"Selected threshold: {v['threshold']:.6f}. Validation sensitivity: {pct(v['sensitivity'])}.", '',
        '| Test measure | Validation-selected threshold | Threshold 0.5 |',
        '| --- | ---: | ---: |']
    for key in ('sensitivity', 'specificity', 'precision', 'negative_predictive_value', 'false_negative_rate'):
        lines.append(f"| {key.replace('_', ' ')} | {pct(t[key])} | {pct(default[key])} |")
    for key in ('tp', 'fn', 'fp', 'tn'):
        lines.append(f'| {key.upper()} | {t[key]} | {default[key]} |')
    lines += ['', f"Test ROC AUC: {t['roc_auc']:.3f}; average precision: {t['average_precision']:.3f}. "
        f"Test fracture prevalence (reference level for average precision): {t['fracture_prevalence']:.3f}.", '',
        f"Test sensitivity Wilson 95% interval: {pct(t['sensitivity_ci95_wilson'][0])} to {pct(t['sensitivity_ci95_wilson'][1])}.", '',
        '## Interpretation', '',
        'Assess sensitivity together with specificity and false positives. A high sensitivity obtained by flagging '
        'most normal images does not demonstrate useful discrimination. The validation sensitivity target is '
        'an empirical operating point and is not a guaranteed test result.', '',
        'The baseline uses coarse image/gradient features and is a reference experiment. CNN training is separate. '
        'There are no subtle-fracture or previously-missed labels; these results cannot establish performance on those groups. '
        'Patient independence is unverified; exact duplicates are grouped but related views may remain. '
        'External evaluation and expert adjudication are still needed.', '',
        '## Error review', '',
        f'[Test predictions](artifacts/{args.model}/test_predictions.csv) · '
        f'[False negatives](artifacts/{args.model}/false_negatives.csv) · '
        f'[Detailed metrics](artifacts/{args.model}/metrics.json)', '',
        f'![ROC and precision-recall curves](artifacts/{args.model}/curves.png)', '',
        'The following review shows HUMAN reference annotations on missed images, not model-predicted locations.', '',
        f'![Missed-fracture reference annotations](artifacts/{args.model}/missed_fractures.png)', '']
    (ROOT / f'RESULTS_{args.model}.md').write_text('\n'.join(lines), encoding='utf-8')

if __name__ == '__main__':
    main()
