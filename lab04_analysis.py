"""Measured tables for the lab's appendix; never imports reference-run counts."""

from collections import Counter, defaultdict
import json

from lab04_core import NAME_FIELDS, error_kind, flatten, leaf, normalize_text, wilson
try:
    from lab04_report import arithmetic_summary, kazakh_confusions, selected_hits
except ModuleNotFoundError:
    # In the self-contained notebook these functions are defined by the preceding cell.
    if 'arithmetic_summary' not in globals():
        raise


def measure(hits):
    hits = list(hits)
    k, n = sum(hits), len(hits)
    lo, hi = wilson(k, n)
    return {'correct': k, 'fields': n, 'accuracy': k/n if n else None,
            'wilson_95': [lo, hi]}


def fmt(value):
    if not value['fields']:
        return 'нет данных'
    k, n = value['correct'], value['fields']
    lo, hi = value['wilson_95']
    return f'{k}/{n}; {100*k/n:.1f}% [{100*lo:.1f}; {100*hi:.1f}]'


def raw_json_object(raw):
    try:
        return isinstance(json.loads(raw), dict)
    except (json.JSONDecodeError, TypeError):
        return False


def analyze(docs, truth, ocr_modes, results, letters):
    summary = {'accuracy':{}, 'ocr_modes':{}, 'dates':{}, 'arithmetic':{}, 'confusions':{}, 'letter_experiment':{}, 'by_field':{}}
    for cond in ('clean','degraded'):
        summary['accuracy'][cond] = {}
        for lang in ('all','kk','ru','en'):
            args = {} if lang=='all' else {'lang':lang}
            summary['accuracy'][cond][lang] = {
                label:measure(selected_hits(docs, records, cond, metric, **args))
                for label,records,metric in [('ocr',ocr_modes[4],'found'),('strict',results,'strict'),('norm',results,'norm')]
            }
            summary['accuracy'][cond][lang]['norm_names'] = measure(
                selected_hits(docs,results,cond,'norm',names=True,**args))
            summary['accuracy'][cond][lang]['norm_other'] = measure(
                ok for d in docs if lang=='all' or d['lang']==lang
                for path,ok in results[(d['id'],cond)]['norm'].items() if leaf(path) not in NAME_FIELDS)
        summary['dates'][cond] = {metric:measure(results[(d['id'],cond)][metric]['date'] for d in docs)
                                  for metric in ('strict','norm')}
        summary['arithmetic'][cond] = arithmetic_summary(docs, results, cond)
        summary['by_field'][cond] = {}
        for label,records,metric in [('ocr',ocr_modes[4],'found'),('strict',results,'strict'),('norm',results,'norm')]:
            grouped = defaultdict(list)
            for d in docs:
                for path,ok in records[(d['id'],cond)][metric].items():
                    grouped[leaf(path)].append(ok)
            summary['by_field'][cond][label] = {field:measure(h) for field,h in grouped.items()}
        summary['letter_experiment'][cond] = {}
        for label,records in [('before',results),('after',letters)]:
            summary['letter_experiment'][cond][label] = {
                'names':measure(selected_hits(docs,records,cond,'norm',lang='kk',names=True)),
                'all':measure(selected_hits(docs,records,cond,'norm',lang='kk')),
                'parsed_objects':sum(records[(d['id'],cond)]['pred'] is not None for d in docs if d['lang']=='kk'),
            }
        paired = Counter()
        for d in docs:
            if d['lang']!='kk':
                continue
            for path,before in results[(d['id'],cond)]['norm'].items():
                if leaf(path) in NAME_FIELDS:
                    after = letters[(d['id'],cond)]['norm'][path]
                    paired['improved' if after and not before else 'regressed' if before and not after else 'unchanged'] += 1
        summary['letter_experiment'][cond]['paired_names'] = dict(paired)
    for mode,records in ocr_modes.items():
        summary['ocr_modes'][str(mode)] = {cond:measure(selected_hits(docs,records,cond,'found')) for cond in ('clean','degraded')}
    for condition in (None,'clean','degraded'):
        pairs = kazakh_confusions(docs,truth,results,condition)
        special = 'ӘәҒғҚқҢңӨөҰұҮүҺһІі'
        filtered = Counter({(a,b):n for (a,b),n in pairs.items() if a in special})
        encode = lambda counts:[{'true':a,'predicted':b,'true_codepoint':f'U+{ord(a):04X}',
                                'predicted_codepoint':f'U+{ord(b):04X}','count':n}
                               for (a,b),n in counts.most_common()]
        summary['confusions'][condition or 'all'] = {'all_letters':encode(pairs),'kazakh_specific':encode(filtered)}
    summary['json'] = {cond:{
        'raw_valid_objects':sum(raw_json_object(results[(d['id'],cond)]['raw']) for d in docs),
        'recovered_objects':sum(results[(d['id'],cond)]['pred'] is not None for d in docs),
        'token_limit_reached':sum(results[(d['id'],cond)].get('metrics',{}).get('generated_tokens',0)>=512 for d in docs),
        'documents':len(docs)} for cond in ('clean','degraded')}
    summary['inference_seconds'] = {'main':sum(r.get('seconds',0) for r in results.values()),
                                    'letters':sum(r.get('seconds',0) for r in letters.values())}
    return summary


def detailed_report(summary, environment, ocr_environment):
    rows = ['# Lab 04 — полные результаты собственного прогона', '',
        'Модель: **'+environment['base_model_id']+'**; экспорт **'+environment['model_id']+'**, ревизия `'+environment['model_revision']+'`.',
        f"Runtime: OpenVINO {environment['versions']['openvino']}, GenAI {environment['versions']['openvino-genai']}; "
        f"устройство {environment['device']} — {environment['device_name']}. Веса: {environment['dtype']}; "
        f"точность вычислений CPU: {environment['activation_precision_hint']}. Greedy, max_new_tokens=512.",
        'Это INT4/INT8-экспорт, а не эталонный float16-прогон преподавателя. Истинные значения не передавались модели.', '',
        'OCR: Tesseract '+ocr_environment['tesseract']+', kaz+rus+eng, tessdata_fast 4.1.0. Сохранённые OCR-измерения использованы повторно.', '',
        '## Точность по языкам и качеству', '',
        'В ячейках: k/n; процент и 95% Wilson. Интервалы описательные: поля одного документа и его переводы зависимы.', '',
        '| Скан | Язык | OCR upper bound | VLM strict | VLM normalised |', '|---|---|---|---|---|']
    for cond,langs in summary['accuracy'].items():
        for lang,methods in langs.items():
            rows.append(f'| {cond} | {lang} | '+ ' | '.join(fmt(methods[m]) for m in ('ocr','strict','norm'))+' |')
    rows += ['', 'OCR upper bound проверяет наличие значений, а не их привязку к полям/строкам. Сравнение с JSON-извлечением VLM не является сравнением двух готовых парсеров.', '',
             '### Названия и остальные поля VLM normalised', '',
             '| Скан | Язык | Названия | Остальные поля |','|---|---|---|---|']
    for cond,langs in summary['accuracy'].items():
        for lang,methods in langs.items():
            rows.append(f"| {cond} | {lang} | {fmt(methods['norm_names'])} | {fmt(methods['norm_other'])} |")
    rows += ['', 'Названия — продавец, покупатель и товары. Разделение помогает проверить, связан ли языковой разрыв с текстом при одинаковых числах; оно само по себе не доказывает причину ошибок.', '',
             '## Формат ответа модели', '',
             '| Скан | JSON-объект без обработки | Разобрано штатным парсером | Достигнут лимит 512 токенов |',
             '|---|---:|---:|---:|']
    for cond,j in summary['json'].items():
        rows.append(f"| {cond} | {j['raw_valid_objects']}/{j['documents']} | {j['recovered_objects']}/{j['documents']} | {j['token_limit_reached']} |")
    rows += ['', 'Парсер допускает Markdown-ограждение и извлечение объекта из текста. Ответы не исправлялись вручную; неразобранный JSON получает нулевые оценки всех полей. Достижение лимита само по себе не доказывает, что ответ оборван.', '',
             '## Core 1: сегментация OCR', '', '| PSM | Clean | Degraded |','|---|---|---|']
    for mode in ('3','4','6','11'):
        rows.append(f"| {mode} | {fmt(summary['ocr_modes'][mode]['clean'])} | {fmt(summary['ocr_modes'][mode]['degraded'])} |")
    rows += ['', 'Для этого шаблона выбираю psm 4: он сохраняет строки таблицы. Более высокий upper bound psm 11 не доказывает лучшую привязку ячеек к строкам.', '',
             '## Core 2: даты', '', '| Скан | Strict | Normalised |','|---|---|---|']
    for cond,values in summary['dates'].items():
        rows.append(f"| {cond} | {fmt(values['strict'])} | {fmt(values['norm'])} |")
    rows += ['', 'Финансовой команде сообщил бы normalised как точность чтения; strict отдельно показывает соблюдение формата. Перед передачей в учётную систему дату нужно проверить и привести к ISO.', '',
             '## Точность по типам полей', '', '| Скан | Поле | OCR | VLM strict | VLM normalised |','|---|---|---|---|---|']
    for cond,methods in summary['by_field'].items():
        for field in methods['ocr']:
            rows.append(f'| {cond} | {field} | '+' | '.join(fmt(methods[m][field]) for m in ('ocr','strict','norm'))+' |')
    rows += ['', '## Core 3: арифметические проверки', '',
             '| Скан | Неверные итоги | Пойманы | Правильные итоги | Отмечены | Документы прошли | Прошли с ошибками |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for cond,a in summary['arithmetic'].items():
        rows.append(f"| {cond} | {a['wrong_totals']} | {a['caught']} | {a['right_totals']} | {a['right_flagged']} | {a['passed']} | {a['passed_with_errors']} |")
    rows += ['', 'Отмеченный правильный итог может сопровождаться неправильными строками; это не обязательно ложная тревога о документе в целом. '
             'Арифметика проверяет согласованность чисел, не их соответствие снимку. Ей нельзя поручить автоматическое одобрение оплаты.', '',
             '## Замены букв в названиях казахских счетов', '',
             'Направление: истина → модель. Регистр сохранён: Қ и қ считаются разными Unicode-символами. Считаются только замены букв одинаковой длины, выделенные SequenceMatcher; вставки/удаления в эту таблицу не входят.', '']
    for cond,groups in summary['confusions'].items():
        rows += [f'### {cond}', '', '| Группа | Истина | Модель | Число |','|---|---|---|---:|']
        for label,key in [('Все буквы (топ-3)','all_letters'),('Казахские специфические (топ-3)','kazakh_specific')]:
            for r in groups[key][:3]:
                rows.append(f"| {label} | {r['true']} ({r['true_codepoint']}) | {r['predicted']} ({r['predicted_codepoint']}) | {r['count']} |")
    rows += ['', 'По заменам нельзя отделить зрительное распознавание от языкового смещения. Нужен контроль с настоящими словами и случайными строками из тех же букв при одинаковых шрифте, размере и качестве; несколько формулировок промпта помогут отделить эффект инструкции.', '',
             '## Advanced 5: сохранять казахские буквы', '',
             '| Скан | Названия до | Названия после | Все поля до | Все поля после | Исправлены / испорчены названия |',
             '|---|---|---|---|---|---|']
    for cond,v in summary['letter_experiment'].items():
        p=v['paired_names']
        rows.append(f"| {cond} | {fmt(v['before']['names'])} | {fmt(v['after']['names'])} | {fmt(v['before']['all'])} | {fmt(v['after']['all'])} | {p.get('improved',0)} / {p.get('regressed',0)} |")
    rows += ['', 'В таблице указаны правильные поля. Неверных названий:']
    for cond,v in summary['letter_experiment'].items():
        before,after=v['before']['names'],v['after']['names']
        rows.append(f"- {cond}: {before['fields']-before['correct']}/{before['fields']} → {after['fields']-after['correct']}/{after['fields']}; разобранные JSON-объекты: {v['before']['parsed_objects']}/8 → {v['after']['parsed_objects']}/8.")
    rows += ['', 'Изменение инструкции может менять и читаемые буквы, и формат/полноту JSON. Оно не доказывает наличие или отсутствие знания языка. Один опыт на восьми счетах не отделяет языковое смещение от оптической ошибки и не доказывает статистическое превосходство.', '',
             '## Выбор системы', '',
             'Контроль качества снимка и повторное сканирование при размытии; OCR с парсером для читаемых шаблонов, VLM для исключений; '
             'валидация JSON, обязательных полей, дат, БИН, строк и итогов; подтверждение человеком перед оплатой.', '',
             '## Воспроизводимость', '',
             'Оригинальные ответы сохранены отдельно от оценок. `run_config.json` фиксирует модель, промпты, версии, CPU, FP32 и manifest. '
             'Атомарные файлы сохраняются после каждого изображения; блокировка исключает второй worker, завершённые изображения пропускаются при возобновлении.', '',
             f"Суммарное время сохранённых вызовов: основной прогон {summary['inference_seconds']['main']:.1f} с; дополнительный {summary['inference_seconds']['letters']:.1f} с.", '',
             '```json', json.dumps(environment,ensure_ascii=False,indent=2), '```', '']
    return '\n'.join(rows)
