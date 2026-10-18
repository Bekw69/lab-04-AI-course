"""Summarise measured Lab 04 results; no reference-run counts are used here."""

from __future__ import annotations

from collections import Counter

from lab04_core import (
    NAME_FIELDS, char_confusions, flatten, leaf, normalize_text, wilson,
)


def accuracy(hits):
    """Format field accuracy and its 95% Wilson interval."""
    hits = list(hits)
    k, n = sum(hits), len(hits)
    if not n:
        return "нет данных"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100*k/n:.1f}% [{100*lo:.1f}; {100*hi:.1f}]"


def selected_hits(docs, records, condition, metric, *, lang=None, names=False):
    for doc in docs:
        if lang is not None and doc["lang"] != lang:
            continue
        for path, ok in records[(doc["id"], condition)][metric].items():
            if not names or leaf(path) in NAME_FIELDS:
                yield ok


def arithmetic_summary(docs, results, condition):
    records = [results[(d["id"], condition)] for d in docs]
    passed = lambda r: all(r["arith"].values())
    wrong = [r for r in records if not r["norm"]["total"]]
    right = [r for r in records if r["norm"]["total"]]
    return {
        "wrong_totals": len(wrong),
        "caught": sum(not passed(r) for r in wrong),
        "right_totals": len(right),
        "right_flagged": sum(not passed(r) for r in right),
        "passed": sum(passed(r) for r in records),
        "passed_with_errors": sum(passed(r) and not all(r["norm"].values()) for r in records),
    }


def kazakh_confusions(docs, truth, results, condition=None):
    pairs = Counter()
    for doc in docs:
        if doc["lang"] != "kk":
            continue
        for cond in ((condition,) if condition else ("clean", "degraded")):
            result = results[(doc["id"], cond)]
            expected, predicted = flatten(truth[doc["id"]]), flatten(result["pred"] or {})
            for path, ok in result["norm"].items():
                got = normalize_text(predicted.get(path))
                if not ok and leaf(path) in NAME_FIELDS and isinstance(got, str):
                    pairs.update(char_confusions(normalize_text(expected[path]), got))
    # Count letter substitutions only; punctuation is not a letter confusion.
    return Counter({(a, b): n for (a, b), n in pairs.items() if a != b and a.isalpha() and b.isalpha()})


def submission_report(docs, truth, ocr, results, letter_results, *, environment=None):
    """A short hand-in generated exclusively from the current run."""
    expected = {(d["id"], c) for d in docs for c in ("clean", "degraded")}
    if set(ocr) != expected or set(results) != expected:
        raise ValueError("Run OCR and the VLM on every image before making the report.")
    kk_ids = {d['id'] for d in docs if d['lang'] == 'kk'}
    kk_keys = {k for k in expected if k[0] in kk_ids}
    if set(letter_results) != kk_keys:
        raise ValueError("Run the letter-preservation experiment on all 16 Kazakh images.")

    rows = [
        "# Lab 04 — Invoice to JSON",
        "",
        "Рабочие ожидания сформулированы после чтения README, до собственного прогона; "
        "это не слепые прогнозы. Все числа ниже рассчитаны из текущих результатов.",
        "",
        "**Predict 1.** Ожидаю больше ошибок в казахских названиях продавца, покупателя и товаров; "
        "ухудшение изображения сильнее повредит OCR. Оценка для clean: OCR около 90%, VLM около 85%.",
        "**Predict 2.** На clean ожидаю преимущество OCR upper bound, на degraded — VLM.",
        "",
        "| Метод; 95% Wilson | Clean | Degraded |",
        "|---|---|---|",
    ]
    if environment:
        rows[0] += ' (OpenVINO INT4, Intel CPU)'
        rows[2:2] = [
            f"Qwen2-VL-2B-Instruct, экспорт {environment['model_id']}; "
            f"OpenVINO {environment['versions']['openvino']}; CPU {environment['device_name']}; "
            f"веса INT4/INT8, вычисления {environment['activation_precision_hint']}. Это собственный прогон квантизованной модели.",
            "",
        ]
    for label, records, metric in (("OCR upper bound", ocr, "found"), ("VLM normalised", results, "norm")):
        cells = [accuracy(selected_hits(docs, records, cond, metric)) for cond in ("clean", "degraded")]
        rows.append(f"| {label} | {' | '.join(cells)} |")

    def rate(records, cond, metric, lang=None):
        hits = list(selected_hits(docs, records, cond, metric, lang=lang))
        return sum(hits) / len(hits)
    clean_ocr, bad_ocr = [rate(ocr, c, "found") for c in ("clean", "degraded")]
    clean_vlm, bad_vlm = [rate(results, c, "norm") for c in ("clean", "degraded")]
    weakest = min(sorted({d['lang'] for d in docs}), key=lambda lang: rate(results, "clean", "norm", lang))
    rows += [
        "",
        f"**Измерение для Predict 1.** На clean слабее всего VLM в {weakest}; "
        f"падение clean→degraded: OCR {(clean_ocr-bad_ocr)*100:.1f} п.п., "
        f"VLM {(clean_vlm-bad_vlm)*100:.1f} п.п.; ошибка ожидания clean: "
        f"OCR {(clean_ocr-.90)*100:+.1f} п.п., VLM {(clean_vlm-.85)*100:+.1f} п.п.",
        f"**Измерение для Predict 2.** На clean больше полей у "
        f"{'OCR upper bound' if clean_ocr > clean_vlm else 'VLM' if clean_vlm > clean_ocr else 'обоих поровну'}, "
        f"на degraded — {'VLM' if bad_vlm > bad_ocr else 'OCR upper bound' if bad_ocr > bad_vlm else 'обоих поровну'}.",
        "",
    ]
    confusions = kazakh_confusions(docs, truth, results).most_common(3)
    pairs = [f"{a} (U+{ord(a):04X}) → {b} (U+{ord(b):04X}): {n}" for (a, b), n in confusions]
    rows += [
        "**Три частые замены букв в kk, обе версии вместе (истина → модель):** "
        + ("; ".join(pairs) if pairs else "замен букв не обнаружено") + ".",
        "По одним заменам нельзя отделить зрительное распознавание от языкового смещения; "
        "для проверки нужно сравнить одинаковые казахские буквы в реальных словах и случайных "
        "строках при неизменных шрифте, размере и качестве изображения.",
        "",
    ]
    for cond in ("clean", "degraded"):
        a = arithmetic_summary(docs, results, cond)
        rows.append(
            f"**Арифметика, {cond}:** поймано {a['caught']}/{a['wrong_totals']} неверных итогов; "
            f"отмечено {a['right_flagged']}/{a['right_totals']} правильных; "
            f"прошли {a['passed']} документов, из них {a['passed_with_errors']} с ошибкой в другом поле."
        )
    rows += [
        "Арифметике нельзя разрешать оплату: согласованные неверные числа и ошибки в названиях "
        "могут пройти обе проверки.",
        "",
        "**Эксперимент с казахскими буквами — неверные названия до → после:**",
    ]
    for cond in ("clean", "degraded"):
        before = list(selected_hits(docs, results, cond, "norm", lang="kk", names=True))
        after = list(selected_hits(docs, letter_results, cond, "norm", lang="kk", names=True))
        rows.append(f"{cond}: {len(before)-sum(before)}/{len(before)} → {len(after)-sum(after)}/{len(after)}.")
    rows += [
        "Этот контроль меняет инструкцию, поэтому сам по себе не доказывает причину ошибок; "
        "нужен также опыт со случайными строками и несколькими формулировками.",
        "",
        "**Решение:** сначала контроль качества и повторный скан при размытии; затем OCR с парсером "
        "для читаемых шаблонов, VLM для исключений, проверка схемы, дат, БИН и арифметики "
        "и подтверждение человеком перед оплатой.",
        "",
        "OCR здесь — наличие значений в тексте, а не готовый JSON. Wilson рассчитан по полям; "
        "поля и языковые версии одного счёта зависимы, поэтому интервалы описательные "
        "и не доказывают превосходство на новых документах.",
    ]
    return "\n".join(rows) + "\n"
