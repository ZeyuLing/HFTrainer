"""Plot measured demo losses without smoothing or claiming convergence."""
import argparse
import csv
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--h3-log', type=Path, required=True)
    parser.add_argument('--ltx-metrics', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = []
    for line in args.h3_log.read_text().splitlines():
        match = re.search(r'step \[(\d+)/\d+\].*?  loss=([\d.eE+-]+)', line)
        if match:
            rows.append(dict(model='h3', split='train', step=int(match[1]), loss=float(match[2])))
        match = re.search(r'step=(\d+)  val/loss=([\d.eE+-]+)', line)
        if match:
            rows.append(dict(model='h3', split='validation_random_noise', step=int(match[1]), loss=float(match[2])))
    for line in args.ltx_metrics.read_text().splitlines():
        row = json.loads(line)
        if 'train/loss' in row:
            rows.append(dict(model='ltx', split='train', step=int(row['train/global_step']), loss=row['train/loss']))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'loss.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['model', 'split', 'step', 'loss'])
        writer.writeheader(); writer.writerows(rows)
    summary = {}
    for model, title in [('h3', 'MiniMax H3'), ('ltx', 'LTX 2.5')]:
        train = [r for r in rows if r['model'] == model and r['split'] == 'train']
        if not train:
            raise ValueError(f'Missing real loss history for {model}')
        fig, ax = plt.subplots(figsize=(8, 3.8), layout='constrained')
        ax.plot([r['step'] for r in train], [r['loss'] for r in train], '.-', label='Training (raw)')
        val = [r for r in rows if r['model'] == model and r['split'] != 'train']
        if val:
            ax.plot([r['step'] for r in val], [r['loss'] for r in val], 's--', label='Held-out (resampled noise)')
        ax.set(title=f'{title}: real-data 20-step LoRA smoke run', xlabel='Optimizer step', ylabel='Loss')
        ax.grid(alpha=.2); ax.legend()
        fig.savefig(args.output / f'{model}_loss.png', dpi=160)
        fig.savefig(args.output / f'{model}_loss.svg')
        plt.close(fig)
        first = sum(r['loss'] for r in train[:5]) / len(train[:5])
        last = sum(r['loss'] for r in train[-5:]) / len(train[-5:])
        summary[model] = dict(points=len(train), first=train[0]['loss'], last=train[-1]['loss'], first5_mean=first, last5_mean=last, convergence_demonstrated=False)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
