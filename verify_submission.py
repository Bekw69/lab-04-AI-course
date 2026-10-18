"""Validate data, notebook sources and saved measurements before submission.

Run with --require-complete after inference and notebook execution. Completed
Intel results are detected automatically; an OCR-only run stays incomplete.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path

import nbformat

from lab04_core import arithmetic_checks, flatten, ocr_found, parse_model_json, score_fields
from lab04_checkpoint import LETTER_PROMPT


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify(root, require_complete=False, results_dir=None):
    manifest = load(root / 'data/manifest.json')
    docs = manifest['docs']
    assert len(docs) == 24
    assert {d['lang'] for d in docs} == {'kk', 'ru', 'en'}
    truth = {}
    for doc in docs:
        for condition, suffix in (('truth', 'json'), ('clean', 'png'), ('degraded', 'jpg')):
            path = root / 'data' / condition / f'{doc["id"]}.{suffix}'
            content = path.read_bytes()
            # A Windows checkout can change JSON line endings, never image bytes.
            if suffix == 'json':
                content = content.replace(b'\r\n', b'\n')
            assert hashlib.sha256(content).hexdigest() == doc['sha256'][condition], path
        value = load(root / 'data/truth' / f'{doc["id"]}.json')
        assert len(flatten(value)) == doc['n_fields']
        assert all(arithmetic_checks(value).values()), doc['id']
        truth[doc['id']] = value
    assert sum(len(flatten(t)) for t in truth.values()) == 456
    for base in {d['base_id'] for d in docs}:
        translations = [truth[d['id']] for d in docs if d['base_id'] == base]
        numeric = [{p: v for p, v in flatten(t).items()
                    if p.rsplit('.', 1)[-1] not in ('seller_name', 'buyer_name', 'name')}
                   for t in translations]
        assert len(numeric) == 3 and numeric[0] == numeric[1] == numeric[2]
    print('Data: 48 image hashes + 24 JSON hashes (LF), 456 fields/condition, identical numbers across languages: OK')

    notebook = nbformat.read(root / 'lab04_invoice_to_json.ipynb', as_version=4)
    nbformat.validate(notebook)
    code = [c for c in notebook.cells if c.cell_type == 'code']
    for cell in code:
        ast.parse(cell.source)
        assert not any(o.output_type == 'error' for o in cell.outputs), cell.id
    for filename in ('lab04_core.py', 'lab04_report.py', 'lab04_intel.py', 'lab04_checkpoint.py', 'lab04_analysis.py'):
        source = (root / filename).read_text(encoding='utf-8').strip()
        assert any(c.source.strip() == source for c in code), f'Notebook source drift: {filename}'
    print('Notebook: schema, Python syntax, embedded source parity, no saved error outputs: OK')

    pending = []
    if results_dir is None:
        results_dir = Path('results/intel') if (root/'results/intel/run_config.json').exists() else Path('results')
    run_dir = root / results_dir
    print('Measured results directory:', results_dir)
    environment = load(run_dir/'environment.json') if (run_dir/'environment.json').exists() else {}
    is_intel = environment.get('backend')=='openvino'
    measured = {}
    expected = {(d['id'], cond) for d in docs for cond in ('clean', 'degraded')}
    for filename in ('ocr_psm3.json', 'ocr_psm4.json', 'ocr_psm6.json', 'ocr_psm11.json',
                     'vlm.json', 'vlm_kk_letters.json'):
        path = run_dir / filename
        if not path.is_file():
            pending.append(filename)
            continue
        records = load(path)
        measured[filename] = {(r['doc_id'],r['condition']):r for r in records}
        keys = {(r['doc_id'], r['condition']) for r in records}
        wanted = {k for k in expected if k[0].endswith('_kk')} if filename == 'vlm_kk_letters.json' else expected
        assert len(records) == len(keys), f'Duplicate image records: {filename}'
        assert keys <= wanted, f'Unexpected image records: {filename}'
        if keys != wanted:
            pending.append(f'{filename}: {len(keys)}/{len(wanted)} images')
        for r in records:
            t = truth[r['doc_id']]
            if filename.startswith('ocr_'):
                assert r['found'] == ocr_found(t, r['text']), filename
            else:
                assert r['pred'] == parse_model_json(r['raw']), filename
                assert r['strict'] == score_fields(t, r['pred'], normalized=False), filename
                assert r['norm'] == score_fields(t, r['pred']), filename
                assert r['arith'] == arithmetic_checks(r['pred']), filename
                if is_intel:
                    from lab04_core import PROMPT
                    prompt = LETTER_PROMPT if filename=='vlm_kk_letters.json' else PROMPT
                    assert r['prompt_sha256']==hashlib.sha256(prompt.encode()).hexdigest(), filename
                    doc = next(d for d in docs if d['id']==r['doc_id'])
                    assert r['image_sha256']==doc['sha256'][r['condition']], filename
                    assert r['device']==environment['device']=='CPU', filename
                    assert r['seconds']>=0 and r['finished_at_utc'], filename
                    stage = 'letters' if filename=='vlm_kk_letters.json' else 'main'
                    journal = load(run_dir/'raw'/stage/f'{r["doc_id"]}_{r["condition"]}.json')
                    assert journal['raw']==r['raw'] and journal['prompt']==prompt
                    assert journal['image_sha256']==r['image_sha256']
        print(f'{filename}: {len(records)} saved images, all scores independently recomputed: OK')
    unexecuted = [c.id for c in code if c.execution_count is None]
    if unexecuted:
        pending.append(f'Notebook: {len(unexecuted)} unexecuted code cells')
    for filename in ('environment.json', 'submission.md'):
        if not (run_dir / filename).is_file():
            pending.append(filename)
    if is_intel:
        config = load(run_dir/'run_config.json')
        for key,value in config.items():
            assert environment[key]==value, f'Environment/config mismatch: {key}'
        assert environment['activation_precision_hint']=='FP32'
        assert environment['quantized'] is True
        if not environment.get('finished_at_utc'):
            pending.append('inference worker completion timestamp')
        for name in ('metrics_summary.json','analysis.md','ocr_environment.json'):
            if not (run_dir/name).exists():
                pending.append(name)
        if not pending:
            from lab04_analysis import analyze, detailed_report
            from lab04_report import submission_report
            modes = {m:measured[f'ocr_psm{m}.json'] for m in (3,4,6,11)}
            calculated = analyze(docs,truth,modes,measured['vlm.json'],measured['vlm_kk_letters.json'])
            assert load(run_dir/'metrics_summary.json')==calculated, 'Aggregate metrics are stale'
            assert (run_dir/'submission.md').read_text(encoding='utf-8')==submission_report(
                docs,truth,modes[4],measured['vlm.json'],measured['vlm_kk_letters.json'],environment=environment)
            assert (run_dir/'analysis.md').read_text(encoding='utf-8')==detailed_report(
                calculated,environment,load(run_dir/'ocr_environment.json'))
            print('Environment, raw journals, all aggregate metrics and both reports: OK')
    if pending:
        print('NOT COMPLETE: ' + '; '.join(pending))
    else:
        print('All required experiments and notebook cells have saved results.')
    if require_complete and pending:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--results-dir', type=Path, default=None)
    args = parser.parse_args()
    verify(Path(__file__).resolve().parent, args.require_complete, args.results_dir)
