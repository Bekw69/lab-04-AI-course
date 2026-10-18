"""Toolbox for Lab 04 -- Invoice to JSON.

This file is the single source of truth for the lab's logic. ``build_notebook.py`` pastes it
verbatim into one notebook cell, and ``reference_run.py`` imports it, so the numbers in the
README come from the same code a student runs. Heavy imports (torch, transformers,
pytesseract) happen inside the functions that need them.
"""

from __future__ import annotations

import difflib
import json
import math
import re
import unicodedata
from collections import Counter
from datetime import date
from typing import Any

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
OCR_LANGS = "kaz+rus+eng"
OCR_PSM = 4  # Tesseract page-segmentation mode: "a single column of text of variable sizes"

# Field names only. No value from any truth file ever reaches a prompt.
PROMPT = (
    "Extract this invoice as JSON with exactly these keys: invoice_number, date (YYYY-MM-DD), "
    "seller_name, seller_bin, buyer_name, items (list of objects with name, qty, unit_price, "
    "amount), total. Numbers must be plain integers. Output JSON only."
)

NUMERIC_FIELDS = ("qty", "unit_price", "amount", "total")
NAME_FIELDS = ("seller_name", "buyer_name", "name")  # "name" = an item name
KAZAKH_FOLD = str.maketrans("ӘәҒғҚқҢңӨөҰұҮүҺһІі", "АаГгКкНнОоУуУуХхИи")
LOOKALIKE_FOLD = str.maketrans("ABCEHKMOPTXYaceopxy", "АВСЕНКМОРТХУасеорху")  # Latin -> identical-looking Cyrillic


# ----------------------------------------------------------------------------- parsing
def parse_model_json(text: str) -> dict[str, Any] | None:
    """Parse a model answer into a dict, tolerating a markdown code fence around it.

    Args:
        text: Raw model output.

    Returns:
        The parsed JSON object, or ``None`` if no JSON object can be recovered.
    """
    body = text.strip()
    body = re.sub(r"^```(?:json)?\s*", "", body)
    body = re.sub(r"\s*```$", "", body)
    for candidate in (body, body[body.find("{"): body.rfind("}") + 1]):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts and lists into ``{"items.0.qty": 40, ...}``."""
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.update(flatten(value, f"{prefix}{key}."))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            out.update(flatten(value, f"{prefix}{i}."))
    else:
        out[prefix.rstrip(".")] = obj
    return out


def leaf(path: str) -> str:
    """Last component of a flattened path: ``items.0.qty`` -> ``qty``."""
    return path.rsplit(".", 1)[-1]


# ----------------------------------------------------------------------------- normalisation
def normalize_date(value: Any) -> Any:
    """Turn ``DD.MM.YYYY``, ``DD/MM/YYYY`` or ``DD-MM-YYYY`` into ISO ``YYYY-MM-DD``.

    Anything else (including an already ISO date) is returned stripped and unchanged.
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    match = re.fullmatch(r"(\d{1,2})[./-](\d{1,2})[./-](\d{4})", text)
    if match:
        day, month, year = (int(g) for g in match.groups())
        try:
            return date(year, month, day).isoformat()
        except ValueError:  # e.g. 31.02.2026: not a real date, leave it for the comparison to fail
            return text
    return text


def normalize_number(value: Any) -> Any:
    """Read ``1 509 000``, ``"1509000"`` or ``1509000.0`` as the integer 1509000.

    Every non-digit character is dropped, so this is deliberately forgiving.
    Returns the input unchanged when no digit is present.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        digits = re.sub(r"\D", "", value)
        if digits:
            return int(digits)
    return value


def normalize_text(value: Any) -> Any:
    """NFC-normalise, collapse whitespace and strip. Case and Kazakh letters are kept."""
    if not isinstance(value, str):
        return value
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value)).strip()


def normalize_field(path: str, value: Any) -> Any:
    """Apply the normaliser that matches the field at ``path``."""
    name = leaf(path)
    if name == "date":
        return normalize_date(value)
    if name in NUMERIC_FIELDS:
        return normalize_number(value)
    return normalize_text(value)


# ----------------------------------------------------------------------------- scoring
def score_fields(truth: dict[str, Any], pred: dict[str, Any] | None, *,
                 normalized: bool = True) -> dict[str, bool]:
    """Field-level exact match of a prediction against the truth.

    Every field in the truth is scored; a missing field or a missing row counts as wrong, an
    extra row in the prediction is ignored. ``normalized=False`` compares raw JSON values.

    Args:
        truth: Ground-truth invoice.
        pred: Parsed model answer, or ``None`` when parsing failed (every field is wrong).
        normalized: Apply ``normalize_field`` to both sides before comparing.

    Returns:
        ``{flattened path: matched}`` for every field of the truth.
    """
    truth_flat = flatten(truth)
    pred_flat = flatten(pred) if pred is not None else {}
    hits: dict[str, bool] = {}
    for path, expected in truth_flat.items():
        got = pred_flat.get(path)
        if normalized:
            expected, got = normalize_field(path, expected), normalize_field(path, got)
        hits[path] = got == expected
    return hits


def arithmetic_checks(pred: dict[str, Any] | None) -> dict[str, bool]:
    """The two checks of Lecture 5.1: qty x price = amount per row, and sum of amounts = total.

    These use only the prediction, never the truth, so they can run in production.
    Unreadable numbers make the check fail.
    """
    if pred is None:
        return {"rows_ok": False, "total_ok": False}
    items = pred.get("items")
    rows_ok, amounts = isinstance(items, list) and len(items) > 0, []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            rows_ok = False
            continue
        qty, price, amount = (normalize_number(item.get(k)) for k in ("qty", "unit_price", "amount"))
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in (qty, price, amount)):
            rows_ok = False
            continue
        amounts.append(amount)
        rows_ok = rows_ok and qty * price == amount
    total = normalize_number(pred.get("total"))
    total_ok = isinstance(total, int) and not isinstance(total, bool) and len(amounts) > 0 and sum(amounts) == total
    return {"rows_ok": bool(rows_ok), "total_ok": bool(total_ok)}


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion (95 % for ``z=1.96``).

    Unlike the plain ``p +- 1.96 * sqrt(p(1-p)/n)`` it stays inside [0, 1] and does not collapse
    to zero width when every trial succeeded: 18/18 gives about (0.82, 1.00).
    """
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# ----------------------------------------------------------------------------- OCR side
def ocr_text(image_path: str, psm: int = OCR_PSM) -> str:
    """Tesseract OCR with Kazakh, Russian and English models.

    Args:
        image_path: Path to the invoice image.
        psm: Tesseract page-segmentation mode. The default (4) reads a ruled table row by row;
            mode 6 treats the page as one block and loses most of the table.
    """
    import pytesseract

    # A path, not a PIL image: pytesseract re-encodes a PIL JPEG (quality 75) before OCR,
    # which would add a second round of compression to the degraded scans.
    return pytesseract.image_to_string(str(image_path), lang=OCR_LANGS, config=f"--psm {psm}")


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip().casefold()


def ocr_found(truth: dict[str, Any], text: str) -> dict[str, bool]:
    """For every truth field: does its value appear in the OCR text at all?

    This is an upper bound for any rule-based parser sitting on top of that OCR text: a field
    that is not in the text cannot be extracted from it. It is not a parse, and a number can
    match by accident (``40`` also occurs inside other numbers' neighbourhoods), so treat small
    differences with care.

    Args:
        truth: Ground-truth invoice.
        text: OCR output for the same image.

    Returns:
        ``{flattened path: found}``.
    """
    squashed = _squash(text)
    found: dict[str, bool] = {}
    for path, value in flatten(truth).items():
        name = leaf(path)
        if name in NUMERIC_FIELDS:
            groups = f"{int(value):,}".split(",")  # 580000 -> ["580", "000"]: found as "580 000" or "580000"
            pattern = r"(?<!\d)" + r"[\s ]?".join(groups) + r"(?!\d)"
            found[path] = re.search(pattern, text) is not None
        elif name == "date":
            shown = date.fromisoformat(str(value)).strftime("%d.%m.%Y")
            found[path] = shown in text
        elif name == "seller_bin":
            found[path] = re.search(rf"(?<!\d){value}(?!\d)", text) is not None
        else:
            found[path] = _squash(str(value)) in squashed
    return found


# ----------------------------------------------------------------------------- VLM side
def pick_device() -> str:
    """Pick an available PyTorch device: CUDA, Intel XPU, Apple MPS, or CPU.

    Intel GPUs without PyTorch XPU support can use the separate OpenVINO runner.
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_vlm(device: str | None = None) -> tuple[Any, Any, str]:
    """Load Qwen2-VL-2B-Instruct in eval mode: float16 on a GPU, float32 on CPU.

    Returns:
        ``(model, processor, device)``.
    """
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    device = device or pick_device()
    dtype = torch.float32 if device == "cpu" else torch.float16
    model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, dtype=dtype).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor, device


def vlm_extract(model: Any, processor: Any, image_path: str, *, prompt: str = PROMPT,
                hint: str | None = None, max_new_tokens: int = 512) -> str:
    """Ask the VLM to read one invoice image. Greedy decoding, so the answer is repeatable.

    Args:
        model: Model from ``load_vlm``.
        processor: Processor from ``load_vlm``.
        image_path: Path to the invoice image.
        prompt: Instruction text.
        hint: Optional OCR text appended to the prompt (the hybrid experiment).
        max_new_tokens: Generation budget.

    Returns:
        The raw decoded answer, before any JSON parsing.
    """
    text = prompt if hint is None else f"{prompt}\n\nOCR text of the same image (may contain errors):\n{hint}"
    if getattr(model, "backend", None) == "openvino":
        return model.extract(image_path, prompt=text, max_new_tokens=max_new_tokens)

    import torch
    from PIL import Image

    with Image.open(image_path) as img:
        messages = [{"role": "user", "content": [{"type": "image", "image": img.convert("RGB")},
                                                    {"type": "text", "text": text}]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_ids = out[0][inputs["input_ids"].shape[1]:]
    return processor.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


# ----------------------------------------------------------------------------- error analysis
def fold_kazakh(text: str) -> str:
    """Replace each Kazakh-specific letter with its nearest plain Cyrillic one (Қ->К, Ә->А, ...)."""
    return text.translate(KAZAKH_FOLD)


def fold_lookalikes(text: str) -> str:
    """Replace Latin letters that look identical to Cyrillic ones (T, O, A, ...) with the Cyrillic letter."""
    return text.translate(LOOKALIKE_FOLD)


def error_kind(truth_text: str, pred_text: Any) -> str:
    """Why is a name wrong? The first kind that explains it, checked in this order.

    * ``missing``: the model returned no string at all
    * ``script``: the same text once Latin look-alike letters are mapped to Cyrillic
      (``TOO`` typed with Latin letters instead of ``ТОО``)
    * ``kazakh``: the same text once Kazakh-specific letters are mapped to plain Cyrillic
    * ``both``: needs both foldings
    * ``misread``: still different, a genuine reading error or invented text
    """
    if not isinstance(pred_text, str):
        return "missing"
    if truth_text == pred_text:
        return "equal"
    if fold_lookalikes(truth_text) == fold_lookalikes(pred_text):
        return "script"
    if fold_kazakh(truth_text) == fold_kazakh(pred_text):
        return "kazakh"
    if fold_lookalikes(fold_kazakh(truth_text)) == fold_lookalikes(fold_kazakh(pred_text)):
        return "both"
    return "misread"


def char_confusions(truth_text: str, pred_text: str) -> Counter[tuple[str, str]]:
    """Count (true char, read char) pairs where the two strings differ by same-length replacement."""
    pairs: Counter[tuple[str, str]] = Counter()
    matcher = difflib.SequenceMatcher(None, truth_text, pred_text, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace" and i2 - i1 == j2 - j1:
            pairs.update(zip(truth_text[i1:i2], pred_text[j1:j2]))
    return pairs
