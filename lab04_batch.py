"""Resumable Intel CPU inference: --all --letters runs all 48 + 16 cases once."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
import traceback

from lab04_checkpoint import LETTER_PROMPT, atomic_json, load_records, recover_raw, run_lock, score_record
from lab04_core import PROMPT, vlm_extract
from lab04_intel import IntelVLM


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('AUTO', 'GPU', 'CPU'), default='CPU')
    parser.add_argument('--doc', default='inv01_en')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--letters', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if Path.cwd().resolve() != root:
        parser.error('Run from the lab-04-AI-course directory.')
    manifest = json.loads(Path('data/manifest.json').read_text(encoding='utf-8'))
    docs = manifest['docs']
    selected = docs if args.all else [d for d in docs if d['id'] == args.doc]
    if not selected:
        parser.error('Unknown document id.')
    truths = {d['id']: json.loads(Path(f'data/truth/{d["id"]}.json').read_text(encoding='utf-8')) for d in docs}
    output = Path('results/intel')
    output.mkdir(parents=True, exist_ok=True)
    experiments = [('main', 'vlm.json' if args.all else f'demo_{args.doc}.json', selected, PROMPT)]
    if args.letters:
        experiments.append(('letters', 'vlm_kk_letters.json', [d for d in docs if d['lang']=='kk'], LETTER_PROMPT))
    log_path = output / 'run_log.jsonl'

    def log(event, **values):
        row = {'time_utc': utc_now(), 'event': event, **values}
        with log_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
        print(json.dumps(row, ensure_ascii=False), flush=True)

    with run_lock():
        stores = {}
        for stage, filename, stage_docs, prompt in experiments:
            expected = {(d['id'], c) for d in stage_docs for c in ('clean','degraded')}
            stores[stage] = load_records(output/filename, truths, expected)
            recover_raw(output/'raw'/stage, stores[stage], truths, expected, prompt, manifest)
            if stores[stage]:
                atomic_json(output/filename, list(stores[stage].values()))
        log('start', completed={stage:len(rows) for stage,rows in stores.items()}, device=args.device)
        started = time.perf_counter()
        model = IntelVLM(args.device)
        fingerprint = model.metadata() | {
            'prompt': PROMPT, 'letter_prompt': LETTER_PROMPT, 'max_new_tokens': 512, 'greedy': True,
            'manifest_sha256': hashlib.sha256(Path('data/manifest.json').read_bytes()).hexdigest(),
            'python': platform.python_version(), 'platform': platform.platform(),
        }
        config_path = output/'run_config.json'
        if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8')) != fingerprint:
            raise ValueError('Run configuration changed; refusing to mix checkpoints.')
        atomic_json(config_path, fingerprint)
        environment = fingerprint | {'started_at_utc': utc_now(), 'load_seconds': round(time.perf_counter()-started,2)}
        atomic_json(output/'environment.json', environment)
        log('model_loaded', device=model.device, name=model.device_name, seconds=environment['load_seconds'])

        for stage, filename, stage_docs, prompt in experiments:
            records = stores[stage]
            total = len(stage_docs)*2
            for doc in stage_docs:
                for condition, suffix in (('clean','png'),('degraded','jpg')):
                    key = doc['id'], condition
                    if key in records:
                        log('resume_skip', stage=stage, doc_id=key[0], condition=condition)
                        continue
                    image_path = Path(f'data/{condition}/{key[0]}.{suffix}')
                    image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
                    if image_hash != doc['sha256'][condition]:
                        raise ValueError(f'Image hash mismatch: {image_path}')
                    log('image_started', stage=stage, doc_id=key[0], condition=condition, completed=len(records), total=total)
                    started = time.perf_counter()
                    try:
                        raw = vlm_extract(model, None, str(image_path), prompt=prompt)
                        elapsed = round(time.perf_counter()-started,2)
                        atomic_json(output/'raw'/stage/f'{key[0]}_{condition}.json', {
                            'doc_id':key[0], 'condition':condition, 'raw':raw, 'seconds':elapsed,
                            'prompt':prompt, 'image_sha256':image_hash, 'finished_at_utc':utc_now(),
                        })
                        record = score_record(*key, raw, truths[key[0]], seconds=elapsed,
                                              device=model.device, image_sha256=image_hash,
                                              prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                                              metrics=model.last_metrics, finished_at_utc=utc_now())
                        records[key] = record
                        atomic_json(output/filename, list(records.values()))
                        log('image_finished', stage=stage, doc_id=key[0], condition=condition,
                            completed=len(records), total=total, seconds=elapsed,
                            normalized_correct=sum(record['norm'].values()), fields=len(record['norm']),
                            parsed_json=record['pred'] is not None)
                    except BaseException as exc:
                        log('error', stage=stage, doc_id=key[0], condition=condition,
                            error=repr(exc), traceback=traceback.format_exc())
                        raise
            log('stage_complete', stage=stage, count=len(records))
        environment['finished_at_utc'] = utc_now()
        atomic_json(output/'environment.json', environment)
        log('complete', completed={stage:len(rows) for stage,rows in stores.items()})


if __name__ == '__main__':
    main()
