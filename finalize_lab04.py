"""Build measured reports only after both CPU experiments are complete."""

import json
from pathlib import Path
import shutil

from lab04_analysis import analyze, detailed_report
from lab04_checkpoint import load_records
from lab04_report import submission_report


def finalize():
    output = Path('results/intel')
    docs = json.loads(Path('data/manifest.json').read_text(encoding='utf-8'))['docs']
    truth = {d['id']:json.loads(Path(f'data/truth/{d["id"]}.json').read_text(encoding='utf-8')) for d in docs}
    expected = {(d['id'],c) for d in docs for c in ('clean','degraded')}
    kk_expected = {k for k in expected if k[0].endswith('_kk')}
    results = load_records(output/'vlm.json',truth,expected)
    letters = load_records(output/'vlm_kk_letters.json',truth,kk_expected)
    if set(results)!=expected or set(letters)!=kk_expected:
        raise RuntimeError(f'Inference is incomplete: {len(results)}/48 main, {len(letters)}/16 letters.')
    environment = json.loads((output/'environment.json').read_text(encoding='utf-8'))
    if 'finished_at_utc' not in environment:
        raise RuntimeError('The inference worker has not finished.')
    modes = {}
    for mode in (3,4,6,11):
        source = Path(f'results/ocr_psm{mode}.json')
        shutil.copyfile(source,output/source.name)
        rows = json.loads(source.read_text(encoding='utf-8'))
        modes[mode] = {(r['doc_id'],r['condition']):r for r in rows}
        if set(modes[mode])!=expected:
            raise RuntimeError('Incomplete OCR cache.')
    shutil.copyfile('results/ocr_environment.json',output/'ocr_environment.json')
    ocr_environment = json.loads(Path('results/ocr_environment.json').read_text(encoding='utf-8'))
    summary = analyze(docs,truth,modes,results,letters)
    (output/'metrics_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (output/'submission.md').write_text(submission_report(docs,truth,modes[4],results,letters,environment=environment),encoding='utf-8')
    (output/'analysis.md').write_text(detailed_report(summary,environment,ocr_environment),encoding='utf-8')
    print('Measured reports written to',output)
    return summary


if __name__=='__main__':
    finalize()
