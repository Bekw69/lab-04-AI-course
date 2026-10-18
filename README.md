# Lab 04 — Invoice to JSON

> **Локальная сдача завершена на Intel CPU / OpenVINO (INT4/INT8, FP32).** Ноутбук содержит фактические выводы; инструкция и собственные результаты — в разделе «Решение в этой рабочей копии» ниже. Разделы Expect описывают отдельный эталон преподавателя. NVIDIA для локальной сдачи не требуется.

Lab session in week 6 (Canvas Assignment 4); content maps to week 5, Multimodal and Image
Generation (Lecture 5.1) — the course is running one week behind its official calendar ·
Narxoz University

A scanned invoice is a picture. A program that books or pays it needs fields: number, date,
seller, BIN, buyer, rows, total. This lab turns pictures into JSON two ways — **classic OCR**
(Tesseract) and a **vision-language model** (Qwen2-VL-2B) — and measures where each one breaks.
The documents come in Kazakh, Russian and English with *identical numbers*, each as a clean
render and as a degraded copy (a bad phone photo), so a difference between languages is not
caused by harder digits.

**No API key. No cost.** Tesseract and one open model run inside the notebook. The model
download is about 4.4 GB, once per Colab session. Every company, BIN and amount is invented, and
every invoice is stamped SAMPLE.

## Setup

Open the notebook in Google Colab — one click:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unreal-kz/lab-04-AI-course/blob/main/lab04_invoice_to_json.ipynb)

1. **Runtime → Change runtime type → T4 GPU.** On a CPU the model is far too slow.
2. Run the first code cell once. It installs Tesseract with its Kazakh and Russian language data,
   and the Python packages Colab does not ship with.
3. From Part 1 on, run cells one at a time and write each prediction down before you run the
   cell that reveals the answer.

The reference run (below) took 267 s for all 48 images on an Apple-silicon Mac (MPS, float16).
A Colab T4 run has not been timed.

## Why these two tools

- **OCR: Tesseract 5 with `kaz+rus+eng`.** The classic baseline: it returns *text*, not fields,
  so a parser would still have to be written on top of it. Part 2 therefore measures something
  generous and simple: does the value of each field appear in the OCR text at all? If it does
  not, no parser could recover it, so that count is an **upper bound** for the whole OCR
  approach. (It is not a parse: a number can match by accident.)
- **VLM: `Qwen/Qwen2-VL-2B-Instruct`.** 2.2 billion parameters, Apache-2.0, downloaded without a
  Hugging Face login in the reference run, about 4.4 GB of weights in float16 (a T4 has 16 GB;
  this has not been run on a T4 yet). It gets the image and one instruction that names the
  fields and contains no values. Decoding is greedy, so the same image gives the same answer
  every time.

## Part 0 — the data and the toolbox

24 invoices (8 invoices × 3 languages), each as a clean PNG and a degraded JPEG: 48 images, 456
fields per condition (invoice number, date, seller name and BIN, buyer name, and for every row
name, quantity, unit price, amount, plus the total). `data/truth/*.json` holds what each invoice
really contains; the models never see it. `lab04_core.py` is the toolbox and is pasted
unchanged into one notebook cell: scoring, OCR, the model call, the arithmetic checks.

## Part 1 — look before you measure

Look at the first invoice, its truth and both pictures. ✏️ **Predict 1**: on which language will
the VLM make the most mistakes, and in which fields? Which loses more when the picture is
degraded? Roughly what percentage will each method get right on the clean images?

## Part 2 — classic OCR

Run Tesseract on all 48 images and count, per field, whether its value appears in the text.
✏️ **Predict 2**: on clean images, which recovers more fields, this OCR upper bound or the VLM?
On degraded images?

## Part 3 — the vision-language model

Download the model, read one invoice with it (clean and degraded) and look at the mistakes: a
wrong digit, a wrong letter, an invented word, a missing field, text that is not valid JSON.
Then run all 48 images.

## Part 4 — what the numbers say

Two scorers: **strict** (the JSON value equals the truth exactly) and **normalised** (dates
converted to `YYYY-MM-DD`, numbers read as integers, spaces tidied). Every percentage comes with
a 95 % Wilson interval; with this few documents the interval is often wide, and two methods whose
intervals overlap are not yet shown to differ. Then: the numbers by field type, a cheap
safety net built from arithmetic (`qty × unit_price = amount`, amounts sum to the total), and the
letter-by-letter confusions in the name fields, with Unicode code points.

## What you hand in

One page. Download the notebook (*File → Download → .ipynb*) after running every cell and
answering every ✏️ prompt, and upload it to Canvas.

1. Your Predict 1 and Predict 2, and one sentence each on what the measurement showed.
2. One small table from Part 4: OCR upper bound and VLM (normalised) accuracy on clean and on
   degraded images, all languages together, with the intervals.
3. The three most frequent letter confusions on Kazakh invoices, and one sentence: is this a
   *reading* problem or a *knowledge of the language* problem? Name an experiment that would tell
   the two apart.
4. The arithmetic checks: how many wrong totals did they catch, how many right totals did they
   flag, and would you let them approve a payment on their own?
5. One sentence: for scanned invoices like these, which pipeline would you ship, and what would
   you put in front of it?

## Extension tasks

Every number under "Expect" is from an actual run (`reference_run.py`, `extra_checks.py`;
Tesseract 5.5.3 with `tessdata_fast` 4.1.0 `kaz`/`rus`/`eng`, `Qwen/Qwen2-VL-2B-Instruct`,
`transformers` 5.18.0, `torch` 2.14.1, MPS float16, greedy decoding, 2026-10-05), not from
memory. Greedy decoding is repeatable on one machine: the notebook and the reference script gave
identical raw text on all 48 images. On Colab, Tesseract comes from the Ubuntu packages, whose
version and language data differ from the reference run, and a T4 computes in a different
numerical order than MPS. Expect the OCR counts and the VLM counts to differ from the ones
below by a few fields; that has not been measured.

### Core — no new tools

**1. Page-segmentation modes.**
Do: run Part 2's OCR with `ocr_text(path, psm=...)` for `psm` 3, 4, 6 and 11 on all 48 images
and count the fields found with `ocr_found`.
Hand in: the 4 × 2 table (mode × clean/degraded), and one sentence on which mode you would use
and why.
Expect (fields found out of 456): clean — psm 3: 426, psm 4: 426 (0.934), psm 6: 193 (0.423),
psm 11: 440 (0.965); degraded — psm 3: 5, psm 4: 3, psm 6: 35 (0.077), psm 11: 33 (0.072).
Upscaling the degraded scans to a minimum width of 1100 or 1650 px before OCR does not rescue
psm 4 either: 3, 7 and 3 fields found for as stored, 1100 and 1650.
Trap: psm 11 scores highest on clean images, but it prints every table cell on its own line,
so "the value is somewhere in the text" no longer says which row it belongs to; psm 4 keeps each
row on one line (look at the OCR text of `inv01_kk` under both). The upper bound gets looser as
the layout information disappears. The default is 4 for that reason: on the two lecture
invoices, used as a development set, modes 3, 4 and 11 all found 18 of 18 fields on the clean
image, so that set could not choose between them. And whatever the mode, OCR on the degraded
scans stays under 0.08.

**2. Strict or normalised?**
Do: compute the date field's accuracy under both scorers, for clean and for degraded images.
Hand in: the two pairs, and one sentence on which scorer you would report to a finance team.
Expect: clean — strict 1/24, normalised 24/24; degraded — strict 4/24, normalised 18/24. Over
all fields: clean 380/456 (0.833) strict vs 403/456 (0.884) normalised; degraded 190/456
(0.417) vs 204/456 (0.447). The whole gap is the date field.
Trap: the prompt asks for `YYYY-MM-DD`, the invoice prints `DD.MM.YYYY`, and on 23 of the 24
clean invoices the model answered in the printed format. It read the same date either way; the
five-point difference between "83 % right" and "88 % right" is a decision about what counts as
an error, made when you pick the scorer, not a property of the model.

**3. What the arithmetic checks can and cannot do.**
Do: from the cell in Part 4, count wrong totals caught, right totals flagged, and documents that
pass the checks but still contain a wrong field.
Hand in: the counts for clean and for degraded.
Expect: clean — 0 wrong totals, 24 right totals, 0 flagged. Degraded — 8 wrong totals, all 8
caught; of 16 right totals 13 flagged; 3 documents pass the checks, and none of those 3 has
every field correct.
Trap: "catches every wrong total" is a statement about totals, not about documents. The checks
only see numbers that have to agree with each other; a misread seller name, or a row whose
quantity and amount are wrong in a consistent way, passes. Used as a trigger for human review
they are useful; used as an approval they are not.

### Advanced — extra credit, pick one

**4. Hybrid: give the model the OCR text.**
Do: call `vlm_extract(..., hint=ocr_text(path))` for all 48 images and score as in Part 4.
Hand in: normalised accuracy per language and condition next to the plain VLM, and the Kazakh
name fields on their own.
Expect (normalised, out of 456 or 152): clean, all languages 403 → 407 (0.884 → 0.893); clean
Kazakh 117 → 128 (0.770 → 0.842); clean Russian 138 → 136; clean English 148 → 143. Kazakh name
fields on clean images: 7/42 → 18/42 (0.167 → 0.429). Degraded, all languages 204 → 216
(0.447 → 0.474); degraded Kazakh name fields stay at 0/42.
Trap: do not read "0.884 → 0.893" as a gain. The hint helped where the OCR text was good and the
model's own reading was weak (Kazakh, clean) and slightly hurt where the model alone was already
best (English). On degraded images the total moved too (mostly Russian, 20 → 29 of 152), yet
that OCR text contains almost none of the field values (3 of 456 found), so the gain cannot be
information from the hint: a longer, different prompt changes the model's behaviour. The
mechanism was not investigated. The two Kazakh-name intervals (0.08–0.31 and 0.29–0.58) nearly
touch: with 42 fields the evidence is suggestive, not settled.

**5. Tell the model to keep the Kazakh letters.**
Do: on the 16 Kazakh images only, append this sentence to the prompt: *"Copy every name letter
by letter, exactly as printed. The text may contain the Kazakh letters ә ғ қ ң ө ұ ү һ і: do not
replace them with Russian letters."*
Hand in: wrong name fields before and after, and one sentence for question 3 of "What you hand
in": what does this result say about *reading* versus *knowing the language*?
Expect: clean Kazakh name fields 7/42 → 10/42 correct, all fields 117/152 → 117/152; degraded
name fields 0/42 → 0/42, all fields 85/152 → 84/152.
Trap: an instruction can ask for a capability; it cannot create one. Three more names right out
of 42 is a small difference on a small sample, and no other phrasing was tried. Compare it with
task 4, where a text that already contains the letters moved the same 42 fields by eleven.

## Files

| File | What it is |
|---|---|
| `lab04_invoice_to_json.ipynb` | 39 cells, 22 executed code cells; Parts 0–4, five answers, Core 1–3 and Advanced 5; actual saved outputs |
| `lab04_core.py` | Toolbox; scoring logic unchanged, Intel/OpenVINO dispatch added; embedded unchanged |
| `data/` | Original 24 invoices × clean/degraded, truth files and manifest |
| `lab04_intel.py`, `requirements-intel.txt` | Fixed-revision OpenVINO INT4/INT8 model and isolated runtime |
| `run_intel.py`, `lab04_batch.py`, `lab04_checkpoint.py` | Single-worker, atomic, resumable 48 + 16 image inference |
| `lab04_report.py`, `lab04_analysis.py` | Short hand-in and complete tables calculated from measured records; embedded unchanged |
| `finalize_lab04.py`, `make_submission_pdf.py` | Rebuild measured reports and the one-page PDF |
| `results/intel/` | Raw model responses, parsed JSON, scores, arithmetic, OCR caches, environment, logs and reports |
| `output/pdf/lab04_submission.pdf` | One-page report for submission |
| `verify_submission.py` | Independent rescoring, hashes, embedded-source parity and complete-notebook checks |
| `tests/` | 21 model-free tests of reports, checkpoint recovery/locking and device selection |
| `make_invoices.py` | Original deterministic invoice generator |

## Решение в этой рабочей копии

Завершены собственный основной VLM-прогон на **48 изображениях** и отдельный эксперимент
на **16 казахских изображениях**. В ноутбуке сохранены фактические выводы всех 22 code-ячеек,
ответы на пять ✏️-заданий, Core 1–3 и Advanced 5. Ожидания честно обозначены как
сформулированные после чтения README, до собственного прогона; это не слепые прогнозы.
Эталонные числа из разделов Expect не подставлялись в измерения.

Основной опыт: `Qwen/Qwen2-VL-2B-Instruct`, экспорт
`helenai/Qwen2-VL-2B-Instruct-ov-int4`, ревизия
`3264f1a456ab58768dbeb514dcb348e4af16c392`.
Веса **INT4** для языковой части и **INT8** для vision/embeddings;
CPU `INFERENCE_PRECISION_HINT=f32` (**FP32**). Greedy, max_new_tokens=512.
Это отдельный квантизованный опыт: runtime и точность отличаются от float16-эталона преподавателя.
Истинные значения использовались только для последующего оценивания.


Окружение: 12th Gen Intel(R) Core(TM) i5-1235U, OpenVINO 2026.4.1, GenAI 2026.4.1.0, Python 3.13.2, Windows-11-10.0.26100-SP0.


| Скан | OCR upper bound; 95% Wilson | VLM normalised; 95% Wilson |
|---|---|---|
| clean | 426/456; 93.4% [90.8; 95.4] | 386/456; 84.6% [81.1; 87.7] |
| degraded | 3/456; 0.7% [0.2; 1.9] | 248/456; 54.4% [49.8; 58.9] |

Полные таблицы по языкам, качеству и типам полей, strict/normalised, ошибки JSON,
Unicode-замены букв, арифметика и дополнительный опыт: `results/intel/analysis.md`.
Краткий отчёт: `results/intel/submission.md`; одностраничная версия: `output/pdf/lab04_submission.pdf`.
Wilson-интервалы описательные: поля и переводы одного документа зависимы.
OCR upper bound — присутствие значения в тексте, а не полноценный парсер полей и строк.

OCR выполнен на всех 48 изображениях в четырёх режимах. Использованы сохранённые
измерения Tesseract 5.5.3.20260724, `tessdata_fast` 4.1.0, `kaz+rus+eng`.

| PSM | Clean, полей из 456 | Degraded, полей из 456 |
|---|---:|---:|
| 3 | 426 | 5 |
| 4 | 426 | 3 |
| 6 | 195 | 35 |
| 11 | 440 | 33 |

### Сохранённые измерения

- `results/intel/vlm.json` — 48 исходных ответов, разобранные объекты, strict/normalised по каждому полю, арифметика и время;
- `results/intel/vlm_kk_letters.json` — 16 ответов с инструкцией сохранять казахские буквы;
- `results/intel/raw/main/`, `raw/letters/` — отдельный атомарный журнал каждого ответа;
- `run_config.json`, `environment.json`, `ocr_environment.json` — точные промпты, ревизия, версии, CPU, точность, время и хеши;
- `run_log.jsonl` — фактический ход вычислений;
- `metrics_summary.json`, `analysis.md`, `submission.md` — измеренные сводные результаты и выводы;
- `ocr_psm3.json`, `ocr_psm4.json`, `ocr_psm6.json`, `ocr_psm11.json` — исходный OCR-текст и оценки;
- `verification.txt` — результаты финальных проверок;
- `lab04_intel_results.zip` — комплект ноутбука, отчётов и результатов без весов и временных файлов.

### Запуск и возобновление на Intel

Проверенное окружение `.tools/intel-venv` уже подготовлено. Для новой установки:

```powershell
python -m venv .tools/intel-venv
.\.tools\intel-venv\Scripts\python.exe -m pip install -r requirements-intel.txt
```

Основной и дополнительный прогоны в одном процессе:

```powershell
.\.tools\intel-venv\Scripts\python.exe -X utf8 -u run_intel.py --device CPU --all --letters
```

Каждый ответ сначала сохраняется отдельно, затем оценивается. При повторном запуске
проверенные готовые изображения пропускаются, а ответ из raw-журнала восстанавливается
без повторной генерации. Блокировка ОС исключает второй worker. Фиксируется конфигурация:
смешать чекпойнты разных промптов, весов или версий нельзя.

`--device AUTO` на данной Iris Xe выбирает проверенный CPU. Контроль GPU с драйвером
32.0.101.5542 дал неверные ответы, а GPU FP32 завершился ошибкой отсутствующего ядра;
явный `--device GPU` для Iris сообщает об этом. Это результат проверки данного компьютера,
а не утверждение обо всех Intel GPU. NVIDIA для выбранного CPU-пути не нужна.

### Ноутбук и Canvas

`lab04_invoice_to_json.ipynb` уже содержит фактические выводы; его можно загружать в Canvas.
При повторном исполнении в проверенном окружении он явно читает завершённые ответы
из `results/intel/` и заново вычисляет таблицы. Тяжёлый прогон сделан worker-ом;
повторная генерация в ячейках не заявляется. Для пересчёта нужен этот каталог результатов.
В Colab без локального cache доступен независимый исходный PyTorch/T4-путь.
GitHub → Colab до commit/push открывает опубликованную версию; для своих выводов
загружайте локальный ноутбук.

Проверки:

```powershell
python -X utf8 -m unittest discover -s tests -v
python -X utf8 verify_submission.py --require-complete
git diff --check
```

Никакой commit/push для этой итерации не выполнялся; существующая история не переписывалась.


Instructor-side, not needed by students and kept in the instructor's course repository rather
than in this one: `build_notebook.py` (generates the notebook),
`reference_run.py` → `reference_run.json`, `run_log.txt` (the recorded run behind the numbers
above), `extra_checks.py` → `extra_checks.json`, `extra_checks_log.txt` (the extension-task
numbers), `check_notebook.py` → `check_notebook_log.txt` (the notebook reproduces the reference
run on all 48 images), `test_lab04_core.py` (12 unit tests of the toolbox, no model needed).
