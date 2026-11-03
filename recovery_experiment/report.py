"""Summarize completed paired cases; partial tables are explicitly marked."""
import argparse
import json
from pathlib import Path
import numpy as np


def summarize(rows):
    summaries = []
    for arm, suite in dict.fromkeys((r['arm'], r['suite']) for r in rows):
        all_rows = [r for r in rows if (r['arm'],r['suite']) == (arm,suite)]
        eligible = [r for r in all_rows if not r.get('excluded')]
        n = len(eligible)
        if not n:
            summaries.append(dict(arm=arm,suite=suite,n=0,excluded=len(all_rows)))
            continue
        item = dict(arm=arm,suite=suite,n=n,excluded=len(all_rows)-n)
        for k in ['success','safe_success']:
            item[k] = sum(r[k] for r in eligible)/n
        for k in ['introduced_collision','introduced_height','reset_collision','reset_height']:
            item[k] = sum(r['safety'][k] for r in eligible)/n
        item['recovery_success'] = item['safe_success'] if suite=='near_boundary' else None
        margins = [r['safety']['min_height_margin'] for r in eligible if r['safety']['min_height_margin'] is not None]
        item['height_margin_p05_m'] = float(np.percentile(margins,5)) if margins else None
        clears = [r['stats']['min_clearance'] for r in eligible]
        item['control_endpoint_ee_clearance_p01_m'] = float(np.percentile(clears,1))
        item['control_endpoint_ee_clearance_under_1cm'] = sum(x<.01 for x in clears)/n
        item['nominal_retention_pp'] = None
        if suite == 'nominal':
            base = {r['seed']:r for r in rows if r['arm']=='A' and r['suite']=='nominal' and not r.get('excluded')}
            paired = [r for r in eligible if r['seed'] in base]
            if paired:
                item['nominal_retention_pp'] = 100*sum(int(r['success'])-int(base[r['seed']]['success']) for r in paired)/len(paired)
        summaries.append(item)
    return summaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default='artifacts/smoke')
    args = ap.parse_args()
    out = Path(args.output)
    source = out/'results.json'
    complete = source.exists()
    if not complete:
        source = out/'results.partial.json'
    rows = json.loads(source.read_text())
    summary = summarize(rows)
    result = dict(complete=complete, source=source.name, summaries=summary,
                  interpretation='Pipeline smoke only; no statistical safety-improvement claim.')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines = ['Completed evaluation' if complete else 'PARTIAL evaluation; pending cases are absent.', '',
             '| Arm | Suite | N | Success | Safe success | Collision | Ceiling | Recovery | Retention pp |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        if not r['n']:
            continue
        rates = [f"{r[k]:.1%}" for k in ['success','safe_success','introduced_collision','introduced_height']]
        recovery = '-' if r['recovery_success'] is None else f"{r['recovery_success']:.1%}"
        retention = '-' if r['nominal_retention_pp'] is None else f"{r['nominal_retention_pp']:+.1f}"
        lines.append(f"| {r['arm']} | {r['suite']} | {r['n']} | " + ' | '.join(rates+[recovery,retention])+' |')
    lines += ['', 'Denominators: eligible episodes per arm and suite. Collision/ceiling columns count policy-introduced events. Reset events and setup exclusions are reported separately in summary.json.', '', result['interpretation']]
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
