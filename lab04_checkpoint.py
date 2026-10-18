"""Atomic, validated checkpoints and a single-process lock for the long CPU run."""

from contextlib import contextmanager
import json
import os
from pathlib import Path

from lab04_core import PROMPT, arithmetic_checks, parse_model_json, score_fields

LETTER_PROMPT = PROMPT + ' ' + (
    'Copy every name letter by letter, exactly as printed. The text may contain the Kazakh '
    'letters ә ғ қ ң ө ұ ү һ і: do not replace them with Russian letters.'
)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextmanager
def run_lock(path=Path('.tools/lab04-inference.lock')):
    """The OS releases the lock even after a crash; do not delete lock files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('Another Lab 04 inference process holds the lock.') from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def score_record(doc_id, condition, raw, truth, **metadata):
    pred = parse_model_json(raw)
    return dict(metadata, doc_id=doc_id, condition=condition, raw=raw, pred=pred,
                strict=score_fields(truth, pred, normalized=False),
                norm=score_fields(truth, pred), arith=arithmetic_checks(pred))


def load_records(path, truths, expected=None):
    path = Path(path)
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding='utf-8'))
    result = {}
    for row in rows:
        key = row['doc_id'], row['condition']
        if key in result or (expected is not None and key not in expected):
            raise ValueError(f'Duplicate or unexpected checkpoint key: {key}')
        scored = score_record(*key, row['raw'], truths[key[0]])
        for metric in ('pred', 'strict', 'norm', 'arith'):
            if row.get(metric) != scored[metric]:
                raise ValueError(f'Checkpoint has inconsistent {metric}: {key}')
        result[key] = row
    return result


def recover_raw(raw_dir, records, truths, expected, prompt, manifest):
    """Recover a finished generation if scoring/checkpointing was interrupted."""
    import hashlib

    expected_hashes = {d['id']: d['sha256'] for d in manifest['docs']}
    for path in sorted(Path(raw_dir).glob('*.json')):
        row = json.loads(path.read_text(encoding='utf-8'))
        key = row['doc_id'], row['condition']
        if key not in expected or key in records:
            continue
        if row['prompt'] != prompt or row['image_sha256'] != expected_hashes[key[0]][key[1]]:
            raise ValueError(f'Raw journal configuration mismatch: {path}')
        records[key] = score_record(*key, row['raw'], truths[key[0]],
            seconds=row['seconds'], device='CPU', image_sha256=row['image_sha256'],
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            finished_at_utc=row['finished_at_utc'], recovered_from_raw=True)
    return records
