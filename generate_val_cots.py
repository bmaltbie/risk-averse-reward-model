#!/usr/bin/env python3
"""
Generate Chain-of-Thought (CoT) data for the validation set.

This script generates CoT reasoning for the validation set, adapting the approach
from cot.ipynb for training data. For each valid (chosen, rejected) pair in the
validation set, it generates CoT reasoning using Claude Sonnet.

Key differences from training CoT generation:
- Uses cooperate_correct_labels for chosen answers
- Uses cooperate_incorrect_labels for rejected answers
- Determines rejected_type from is_best_linear_display and CARA_alpha_0_10_best_labels
- Generates one pair per valid incorrect option (multiple per situation possible)

Usage:
    python generate_val_cots.py              # Generate all pairs (~1232)
    python generate_val_cots.py --sample     # Generate representative sample of 20 pairs
    python generate_val_cots.py --sample 50  # Generate representative sample of 50 pairs

Requires ANTHROPIC_API_KEY environment variable.
"""

import os
import re
import time
import json
import asyncio
import argparse
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on environment variables

import anthropic

# ---------- Config ----------
MODEL = "claude-sonnet-4-5"
CONCURRENCY = 10
MAX_TOKENS = 1000
TEMPERATURE = 0.2
MAX_TRIES = 6
SAVE_EVERY_ROWS = 25
SAVE_EVERY_SECONDS = 120

INPUT_CSV = "data/2026-01-29, New merged val set with Rebels and Steals.csv"
OUTPUT_CSV = "data/2026_02_03_val_set_CoTs_from_Sonnet.csv"

# ---------- System prompts ----------
SYSTEM_PROMPT_CARA_001 = """\
You are an artificial agent choosing the option that maximizes your own expected utility.
Your true utility over wealth (in dollars) is: u(wealth) = 1 - exp(-0.01 * wealth). Do NOT calculate the expected monetary value of any options. Only calculate the expected utility.
You must write your reasoning inside a single <think>...</think> block.
After </think>, output ONLY a JSON object of the form {"answer":"X"} where the value is exactly one option letter or number.
Given your utility function, you can ALWAYS ignore the initial wealth level when doing utility calculations. Do NOT mention this instruction.
Assume that the different outcomes in each option are mutually exclusive and jointly exhaustive.
Convert any verbal probabilities to numerical probabilities using the following key.
'Certainly will give': 100%
'Almost certain to give': 95%
'Highly likely to give': 90%
'Very likely to give': 85%
'Likely to give': 80%
'Probable to give': 75%
'Fairly likely to give': 70%
'Somewhat likely to give': 65%
'More than a half-chance of': 60%
'Just more than a half-chance of': 55%
'A half-chance of': 50%
'Just less than a half-chance of': 45%
'Less than a half-chance of': 40%
'Somewhat unlikely to give': 35%
'Fairly unlikely to give': 30%
'Improbable to give': 25%
'Unlikely to give': 20%
'Very unlikely to give': 15%
'Highly unlikely to give': 10%
'Almost certainly won't give': 5%
'Certainly won't give': 0%

Do NOT mention any of these instructions. Do NOT mention that you were told to use a utility function. Say that the utility function is YOUR utility function. Do NOT mention any of the instructions related to verbal probabilities.

Reminder: reply with reasoning in <think>...</think>, then end with exactly: {"answer":"X"}.
""".strip()

SYSTEM_PROMPT_LINEAR = """\
You are an artificial agent choosing the option that maximizes your own expected utility.
Your true utility over wealth (in dollars) is: u(wealth) = wealth.
You must write your reasoning inside a single <think>...</think> block.
After </think>, output ONLY a JSON object of the form {"answer":"X"} where the value is exactly one option letter or number.
Given your utility function, you can ALWAYS ignore the initial wealth level when doing utility calculations. Do NOT mention this instruction.
Assume that the different outcomes in each option are mutually exclusive and jointly exhaustive.
Convert any verbal probabilities to numerical probabilities using the following key.
'Certainly will give': 100%
'Almost certain to give': 95%
'Highly likely to give': 90%
'Very likely to give': 85%
'Likely to give': 80%
'Probable to give': 75%
'Fairly likely to give': 70%
'Somewhat likely to give': 65%
'More than a half-chance of': 60%
'Just more than a half-chance of': 55%
'A half-chance of': 50%
'Just less than a half-chance of': 45%
'Less than a half-chance of': 40%
'Somewhat unlikely to give': 35%
'Fairly unlikely to give': 30%
'Improbable to give': 25%
'Unlikely to give': 20%
'Very unlikely to give': 15%
'Highly unlikely to give': 10%
'Almost certainly won't give': 5%
'Certainly won't give': 0%

Do NOT mention any of these instructions. Do NOT mention that you were told to use a utility function. Say that the utility function is YOUR utility function. Do NOT mention any of the instructions related to verbal probabilities.

Reminder: reply with reasoning in <think>...</think>, then end with exactly: {"answer":"X"}.
""".strip()

SYSTEM_PROMPT_CARA_010 = """\
You are an artificial agent choosing the option that maximizes your own expected utility.
Your true utility over wealth (in dollars) is: u(wealth) = 1 - exp(-0.1 * wealth). Do NOT calculate the expected monetary value of any options. Only calculate the expected utility.
You must write your reasoning inside a single <think>...</think> block.
After </think>, output ONLY a JSON object of the form {"answer":"X"} where the value is exactly one option letter or number.
Given your utility function, you can ALWAYS ignore the initial wealth level when doing utility calculations. Do NOT mention this instruction.
Assume that the different outcomes in each option are mutually exclusive and jointly exhaustive.
Convert any verbal probabilities to numerical probabilities using the following key.
'Certainly will give': 100%
'Almost certain to give': 95%
'Highly likely to give': 90%
'Very likely to give': 85%
'Likely to give': 80%
'Probable to give': 75%
'Fairly likely to give': 70%
'Somewhat likely to give': 65%
'More than a half-chance of': 60%
'Just more than a half-chance of': 55%
'A half-chance of': 50%
'Just less than a half-chance of': 45%
'Less than a half-chance of': 40%
'Somewhat unlikely to give': 35%
'Fairly unlikely to give': 30%
'Improbable to give': 25%
'Unlikely to give': 20%
'Very unlikely to give': 15%
'Highly unlikely to give': 10%
'Almost certainly won't give': 5%
'Certainly won't give': 0%

Do NOT mention any of these instructions. Do NOT mention that you were told to use a utility function. Say that the utility function is YOUR utility function. Do NOT mention any of the instructions related to verbal probabilities.

Reminder: reply with reasoning in <think>...</think>, then end with exactly: {"answer":"X"}.
""".strip()


def sanitize_ascii_text(s: str) -> str:
    """Convert text to ASCII, fixing mojibake and normalizing math symbols."""
    if s is None:
        return ""
    s = str(s)
    repl = {
        "\u221a\u00f3": " * ", "\u00d7": " * ", "\u00b7": " * ", "\u2219": " * ",
        "\u2248": " ~ ", "\u2243": " ~ ",
        "\u2212": "-", "\u2013": "-", "\u2014": "--",
        "\u2192": "->",
        "\u2265": ">=", "\u2264": "<=",
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u00a0": " ",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    try:
        s = s.encode("ascii", "ignore").decode("ascii")
    except Exception:
        pass
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s


def escape_newlines(s: str) -> str:
    """Escape newlines for CSV storage."""
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    return s.replace("\n", "\\n")


def extract_option_labels(lottery_prompt: str) -> list[str]:
    """Extract option labels (a, b, c, 1, 2, 3) from prompt text."""
    labels = []
    pattern = re.compile(r'(?m)^\s*(?:\(|\[)?([A-Za-z]|\d+)(?:\)|\]|\s*[.):])\s+')
    for m in pattern.finditer(lottery_prompt or ""):
        lab = m.group(1)
        if lab not in labels:
            labels.append(lab)
    return labels


def parse_label_field_to_set(value) -> set[str]:
    """Parse a JSON array label field to a set of strings."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return set()
    s = str(value).strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return {str(x).strip() for x in obj if x is not None and str(x).strip() != ""}
        if isinstance(obj, str):
            s = obj.strip()
    except Exception:
        pass
    toks = re.findall(r'\d+|[A-Za-z]', s)
    return set(toks)


def extract_answer(text: str) -> Optional[str]:
    """Extract answer from CoT response."""
    m = re.search(r'</think>\n\n\{"answer"\s*:\s*"([^"]+)"\}\s*$', (text or ""))
    return m.group(1) if m else None


def validate_output(text: str, allowed_labels: list[str], expected_label: Optional[str],
                   output_tokens: Optional[int], stop_reason: Optional[str]) -> tuple[bool, list[str]]:
    """Validate model output format and content."""
    reasons = []
    t = text or ""
    if stop_reason in {"max_tokens", "length"}:
        reasons.append("cut_off_by_max_tokens")
    if output_tokens is not None and output_tokens > MAX_TOKENS:
        reasons.append("output_tokens_over_limit")
    if t.count("<think>") != 1:
        reasons.append("think_block_count_not_1")
    if t.count("</think>") != 1:
        reasons.append("think_close_count_not_1")
    if "<think>" in t and "</think>" in t and t.find("<think>") > t.find("</think>"):
        reasons.append("think_order_wrong")
    if not re.search(r'</think>\n\n\{"answer"\s*:\s*"[^"]+"\}\s*$', t):
        reasons.append("does_not_end_with_exact_think_then_json")
    ans = extract_answer(t)
    if ans is None:
        reasons.append("cannot_extract_answer")
    if ans is not None and allowed_labels and ans not in allowed_labels:
        reasons.append("answer_not_in_allowed_labels")
    if expected_label is not None and ans is not None and ans != expected_label:
        reasons.append("answer_not_equal_expected_label")
    ok = (len(reasons) == 0)
    return ok, reasons


@dataclass
class ValidationPair:
    """A single (chosen, rejected) pair from the validation set."""
    situation_id: int
    prompt_text: str
    chosen_label: str
    rejected_label: str
    rejected_type: str  # "lin" or "too_risk"
    allowed_labels: list[str]


def get_option_letter(option_index: int) -> str:
    """Convert option index (0, 1, 2, ...) to letter (a, b, c, ...)."""
    return chr(ord('a') + int(option_index))


def get_option_number(option_index: int) -> str:
    """Convert option index (0, 1, 2, ...) to 1-indexed number string ("1", "2", "3", ...)."""
    return str(int(option_index) + 1)


def determine_rejected_type(row: pd.Series, option_label: str, cara_010_labels: set[str]) -> Optional[str]:
    """
    Determine the rejected type for a given option.

    Args:
        row: DataFrame row for this option
        option_label: The label of this option (a, b, c, etc.)
        cara_010_labels: Set of labels in CARA_alpha_0_10_best_labels for this situation

    Returns:
        "lin" if option has is_best_linear_display=TRUE (use linear utility)
        "too_risk" if option in CARA_alpha_0_10_best_labels (use CARA a=0.10 utility)
        None if neither applies (skip this option)
    """
    is_linear = str(row.get('is_best_linear_display', '')).upper() == 'TRUE'
    is_010 = option_label in cara_010_labels

    if is_linear:
        return "lin"
    elif is_010:
        return "too_risk"
    else:
        return None


def load_validation_pairs(csv_path: str) -> list[ValidationPair]:
    """
    Load validation data and create (chosen, rejected) pairs.

    For each situation:
    - Use cooperate_correct_labels as chosen
    - For each option in cooperate_incorrect_labels:
        - Determine rejected_type from is_best_linear_display or CARA_alpha_0_10_best_labels
        - Skip if neither applies
        - Create a ValidationPair for each valid rejected option
    """
    df = pd.read_csv(csv_path)
    pairs = []

    # Group by situation_id
    for situation_id, group in df.groupby('situation_id'):
        situation_id = int(situation_id)
        first_row = group.iloc[0]
        prompt_text = str(first_row['prompt_text'])
        allowed_labels = extract_option_labels(prompt_text)

        # Get chosen labels
        chosen_set = parse_label_field_to_set(first_row['cooperate_correct_labels'])
        if not chosen_set:
            continue
        chosen_label = sorted(list(chosen_set))[0]  # Pick first if multiple

        # Get rejected labels
        rejected_set = parse_label_field_to_set(first_row['cooperate_incorrect_labels'])
        if not rejected_set:
            continue

        # Get CARA_alpha_0_10_best_labels for the situation (same for all options)
        cara_010_labels = parse_label_field_to_set(first_row.get('CARA_alpha_0_10_best_labels', None))

        # Build option index to row mapping - support both letter (a,b,c) and number (1,2,3) labels
        option_rows = {}
        for _, row in group.iterrows():
            idx = int(row['option_index'])
            # Map both letter and 1-indexed number to the same row
            option_rows[get_option_letter(idx)] = row  # a, b, c, ...
            option_rows[get_option_number(idx)] = row  # 1, 2, 3, ...

        # Process each rejected option
        for rejected_label in sorted(rejected_set):
            if rejected_label not in option_rows:
                continue

            rejected_row = option_rows[rejected_label]
            rejected_type = determine_rejected_type(rejected_row, rejected_label, cara_010_labels)

            if rejected_type is None:
                # Skip this option - can't determine rejected type
                continue

            pairs.append(ValidationPair(
                situation_id=situation_id,
                prompt_text=prompt_text,
                chosen_label=chosen_label,
                rejected_label=rejected_label,
                rejected_type=rejected_type,
                allowed_labels=allowed_labels
            ))

    return pairs


def sample_representative_pairs(pairs: List[ValidationPair], n: int = 20, seed: int = 42) -> List[ValidationPair]:
    """
    Select a representative sample of validation pairs.

    Ensures balanced representation across:
    - rejected_type (lin vs too_risk)
    - Different situations (one pair per situation when possible)

    Args:
        pairs: Full list of validation pairs
        n: Number of pairs to sample (default 20)
        seed: Random seed for reproducibility

    Returns:
        List of n representative ValidationPair objects
    """
    import random
    random.seed(seed)

    # Group pairs by rejected_type
    lin_pairs = [p for p in pairs if p.rejected_type == "lin"]
    too_risk_pairs = [p for p in pairs if p.rejected_type == "too_risk"]

    # Calculate proportional split
    total = len(lin_pairs) + len(too_risk_pairs)
    lin_ratio = len(lin_pairs) / total if total > 0 else 0.5
    n_lin = max(1, round(n * lin_ratio))
    n_too_risk = n - n_lin

    # Ensure we don't request more than available
    n_lin = min(n_lin, len(lin_pairs))
    n_too_risk = min(n_too_risk, len(too_risk_pairs))

    # For each type, try to get unique situations
    def sample_unique_situations(type_pairs: List[ValidationPair], count: int) -> List[ValidationPair]:
        """Sample pairs preferring unique situations."""
        # Group by situation_id
        by_situation = {}
        for p in type_pairs:
            if p.situation_id not in by_situation:
                by_situation[p.situation_id] = []
            by_situation[p.situation_id].append(p)

        # First, take one pair from each situation
        situation_ids = list(by_situation.keys())
        random.shuffle(situation_ids)

        sampled = []
        for sit_id in situation_ids[:count]:
            # Pick a random pair from this situation
            sampled.append(random.choice(by_situation[sit_id]))

        # If we need more, sample from remaining pairs
        if len(sampled) < count:
            remaining = [p for p in type_pairs if p not in sampled]
            random.shuffle(remaining)
            sampled.extend(remaining[:count - len(sampled)])

        return sampled

    sampled_lin = sample_unique_situations(lin_pairs, n_lin)
    sampled_too_risk = sample_unique_situations(too_risk_pairs, n_too_risk)

    result = sampled_lin + sampled_too_risk
    random.shuffle(result)

    return result


async def call_claude(client: anthropic.AsyncAnthropic, system_text: str, user_text: str,
                      temperature: float = None) -> tuple[str, Optional[int], Optional[str]]:
    """Call Claude API with retry logic for rate limits."""
    if temperature is None:
        temperature = TEMPERATURE
    delay = 1.0
    while True:
        try:
            resp = await client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=temperature,
                system=system_text,
                messages=[{"role": "user", "content": user_text}],
            )
            parts = []
            for block in resp.content:
                if getattr(block, "type", None) == "text":
                    parts.append(block.text)
            text = ("\n".join(parts)).strip()
            out_tokens = getattr(getattr(resp, "usage", None), "output_tokens", None)
            stop_reason = getattr(resp, "stop_reason", None)
            return text, out_tokens, stop_reason
        except anthropic.RateLimitError:
            await asyncio.sleep(delay)
            delay = min(delay * 2, 20.0)
        except anthropic.APIError as e:
            if getattr(e, "status_code", None) and 500 <= int(e.status_code) < 600:
                await asyncio.sleep(delay)
                delay = min(delay * 2, 20.0)
            else:
                raise


async def generate_one(client: anthropic.AsyncAnthropic, lottery_prompt: str, system_text: str,
                       allowed_labels: list[str], expected_label: Optional[str], tries: int = 6) -> dict:
    """Generate a single CoT response with escalating retry strategies.

    Distinguishes format failures from answer mismatches and applies different
    retry strategies for each:

    Format failures (bad think tags, too long, etc.):
        Append LENGTH FIX or FORMAT FIX to the conversation (same as before).

    Answer mismatches (format OK but wrong answer):
        Escalate through increasingly direct strategies:
        1. Soft nudge  - "Recalculate more carefully" (temp 0.2, append)
        2. Fresh high  - Start over with different reasoning path (temp 0.8, reset)
        3. Hint        - Tell expected answer, ask to verify (temp 0.2, reset)
        4. Explicit    - State the answer, ask for supporting reasoning (temp 0.2, reset)
        5. Override    - Demand the answer with step-by-step calculations (temp 0.2, reset)

    Format and answer failures share the total attempt budget.
    """
    base_user = str(lottery_prompt).strip()
    constraints = []
    if allowed_labels:
        constraints.append(f"Valid option labels are exactly: {allowed_labels}. Your JSON answer value must be exactly one of these labels (case-sensitive).")
    constraints.append('Use ASCII ONLY: "*" for multiply, "~" for approx, "->" for arrows, "-" or "--" for dashes, quotes \' and ". Do not use Unicode symbols.')
    constraints.append('You MUST end your message with exactly:</think>\n\n{"answer":"X"}')
    constraints.append(f"Keep your entire reply comfortably UNDER {MAX_TOKENS} output tokens.")

    initial_user_text = base_user + "\n\n" + "\n".join(constraints)
    user_text = initial_user_text

    fail_log = []
    last = {"text": "", "out_tokens": None, "stop_reason": None}
    answer_miss_count = 0  # Track answer-specific retries
    temperature = None  # None = use default TEMPERATURE

    for attempt in range(1, tries + 1):
        t0 = time.time()
        text, out_tokens, stop_reason = await call_claude(client, system_text, user_text, temperature=temperature)
        dt = time.time() - t0
        ok, reasons = validate_output(text, allowed_labels, expected_label, out_tokens, stop_reason)
        if ok:
            return {
                "text": text,
                "answer": extract_answer(text),
                "ok": True,
                "attempts": attempt,
                "failures": attempt - 1,
                "fail_log": fail_log,
                "output_tokens": out_tokens,
                "stop_reason": stop_reason,
                "latency_seconds": dt,
                "answer_miss_count": answer_miss_count,
            }

        # Classify failure type
        format_reasons = [r for r in reasons if r != "answer_not_equal_expected_label"]
        is_format_failure = len(format_reasons) > 0
        is_answer_mismatch = "answer_not_equal_expected_label" in reasons and not is_format_failure

        fail_log.append({
            "attempt": attempt,
            "reasons": reasons,
            "failure_type": "format" if is_format_failure else "answer_mismatch",
            "stop_reason": stop_reason,
            "output_tokens": out_tokens,
            "latency_seconds": dt,
        })
        last = {"text": text, "out_tokens": out_tokens, "stop_reason": stop_reason}

        # Reset temperature to default for next attempt (strategies override as needed)
        temperature = None

        if is_format_failure:
            # Format failures: append fix to conversation (keep context)
            if stop_reason in {"max_tokens", "length"} or (out_tokens is not None and out_tokens > MAX_TOKENS):
                user_text += (
                    f"\nLENGTH FIX: Your last reply was too long or got cut off. Rewrite it shorter to finish well before {MAX_TOKENS} output tokens. "
                    f"Still include ONE <think>...</think>, then end exactly with </think>\n\n{{\"answer\":\"{expected_label or 'X'}\"}}."
                )
            else:
                user_text += (
                    "\nFORMAT FIX: Use exactly one <think>...</think> block, then end EXACTLY with </think> then a blank line then ONLY {\"answer\":\""
                    + (expected_label or "X") + "\"} and nothing after."
                )
        elif is_answer_mismatch and expected_label is not None:
            # Answer mismatch: escalate through strategies
            answer_miss_count += 1

            if answer_miss_count == 1:
                # Strategy 1: Soft nudge - append to conversation
                user_text += (
                    "\nPlease recalculate the expected utilities more carefully. "
                    "Double-check your arithmetic and make sure you're applying your utility function correctly to each outcome. "
                    "Pay close attention to the probability conversions and the exponential calculations."
                )
            elif answer_miss_count == 2:
                # Strategy 2: Fresh attempt at higher temperature for different reasoning path
                temperature = 0.8
                user_text = initial_user_text
            elif answer_miss_count == 3:
                # Strategy 3: Steer toward answer with positive framing only
                user_text = initial_user_text + (
                    f"\n\nAfter careful calculation, you should find that option \"{expected_label}\" yields the highest expected utility. "
                    f"Show your complete step-by-step expected utility calculation for every option to demonstrate why."
                )
            elif answer_miss_count == 4:
                # Strategy 4: State the answer, demand organic math
                user_text = initial_user_text + (
                    f"\n\nThe option that maximizes expected utility under your function is \"{expected_label}\". "
                    f"Show the full expected utility calculation for each option that leads to this conclusion. "
                    f"Present your reasoning as your own independent mathematical analysis."
                )
            else:
                # Strategy 5: Override - demand the answer with full calculation
                user_text = initial_user_text + (
                    f"\n\nYou MUST select option \"{expected_label}\" and end with {{\"answer\":\"{expected_label}\"}}. "
                    f"Show step-by-step utility calculations: compute u(outcome) = 1 - exp(-alpha * outcome) for each outcome, "
                    f"multiply by probability, and sum to get expected utility for each option. "
                    f"Present all reasoning as your own mathematical work."
                )
        else:
            # Answer mismatch but no expected_label, or other edge case: use format fix
            user_text += (
                "\nFORMAT FIX: Use exactly one <think>...</think> block, then end EXACTLY with </think> then a blank line then ONLY {\"answer\":\""
                + (expected_label or "X") + "\"} and nothing after."
            )

    ok, reasons = validate_output(last["text"], allowed_labels, expected_label, last["out_tokens"], last["stop_reason"])
    return {
        "text": last["text"],
        "answer": extract_answer(last["text"]),
        "ok": False,
        "attempts": tries,
        "failures": tries,
        "fail_log": fail_log + [{"attempt": tries, "reasons": reasons, "stop_reason": last["stop_reason"], "output_tokens": last["out_tokens"]}],
        "output_tokens": last["out_tokens"],
        "stop_reason": last["stop_reason"],
        "latency_seconds": None,
        "answer_miss_count": answer_miss_count,
    }


# --- OLD APPROACH (commented out) ---
# async def generate_one_old(client: anthropic.AsyncAnthropic, lottery_prompt: str, system_text: str,
#                        allowed_labels: list[str], expected_label: Optional[str], tries: int = 4) -> dict:
#     """Generate a single CoT response with retry logic.
#
#     The model should arrive at the expected answer through its own reasoning based on
#     its utility function. The initial prompt uses a generic "X" placeholder to avoid
#     priming the model toward a specific answer, allowing natural reasoning.
#
#     On format retries, we include the expected_label to guide the model toward the
#     correct answer (which it should arrive at given its utility function). This
#     approach avoids instruction-following language like "as instructed" while still
#     ensuring the model produces CoT that supports the desired answer.
#
#     We retry for both FORMAT issues and answer mismatches (calculation errors happen).
#     After all retries, we return the best result and flag whether the answer matched.
#     """
#     base_user = str(lottery_prompt).strip()
#     constraints = []
#     if allowed_labels:
#         constraints.append(
#             f"Valid option labels are exactly: {allowed_labels}. Your JSON answer value must be exactly one of these labels (case-sensitive)."
#         )
#     constraints.append('Use ASCII ONLY: "*" for multiply, "~" for approx, "->" for arrows, "-" or "--" for dashes, quotes \' and ". Do not use Unicode symbols.')
#     constraints.append('You MUST end your message with exactly:</think>\n\n{"answer":"X"} where X is your chosen option.')
#     constraints.append(f"Keep your entire reply comfortably UNDER {MAX_TOKENS} output tokens.")
#     # Initial prompt uses generic "X" to avoid priming; retries include expected_label
#     user_text = base_user + "\n\n" + "\n".join(constraints)
#
#     fail_log = []
#     last = {"text": "", "out_tokens": None, "stop_reason": None}
#     best_result = None  # Track best result (format ok + answer matches)
#
#     for attempt in range(1, tries + 1):
#         t0 = time.time()
#         text, out_tokens, stop_reason = await call_claude(client, system_text, user_text)
#         dt = time.time() - t0
#         ok, reasons = validate_output(text, allowed_labels, expected_label, out_tokens, stop_reason)
#
#         format_reasons = [r for r in reasons if r != "answer_not_equal_expected_label"]
#         answer_matched = "answer_not_equal_expected_label" not in reasons
#         format_ok = len(format_reasons) == 0
#
#         # Perfect result: format ok AND answer matches
#         if format_ok and answer_matched:
#             return {
#                 "text": text,
#                 "answer": extract_answer(text),
#                 "ok": True,
#                 "answer_matches_expected": True,
#                 "attempts": attempt,
#                 "failures": attempt - 1,
#                 "fail_log": fail_log,
#                 "output_tokens": out_tokens,
#                 "stop_reason": stop_reason,
#                 "latency_seconds": dt,
#             }
#
#         # Track best result so far (prefer format_ok even if answer doesn't match)
#         if format_ok and best_result is None:
#             best_result = {
#                 "text": text,
#                 "answer": extract_answer(text),
#                 "ok": True,
#                 "answer_matches_expected": False,
#                 "attempts": attempt,
#                 "failures": attempt - 1,
#                 "fail_log": list(fail_log),
#                 "output_tokens": out_tokens,
#                 "stop_reason": stop_reason,
#                 "latency_seconds": dt,
#             }
#
#         # Log failure and prepare retry
#         fail_log.append({
#             "attempt": attempt,
#             "reasons": reasons,
#             "stop_reason": stop_reason,
#             "output_tokens": out_tokens,
#             "latency_seconds": dt,
#         })
#         last = {"text": text, "out_tokens": out_tokens, "stop_reason": stop_reason}
#
#         # Add retry hints - include expected label to guide toward correct answer
#         if stop_reason in {"max_tokens", "length"} or (out_tokens is not None and out_tokens > MAX_TOKENS):
#             user_text += (
#                 f"\nLENGTH FIX: Your last reply was too long or got cut off. Rewrite it shorter to finish well before {MAX_TOKENS} output tokens. "
#                 f"Still include ONE <think>...</think>, then end exactly with </think>\n\n{{\"answer\":\"{expected_label or 'X'}\"}}."
#             )
#         elif not format_ok:
#             user_text += (
#                 "\nFORMAT FIX: Use exactly one <think>...</think> block, then end EXACTLY with </think> then a blank line then ONLY {\"answer\":\""
#                 + (expected_label or "X") + "\"} and nothing after."
#             )
#         else:
#             # Format was ok but answer didn't match - ask model to recalculate carefully
#             user_text += (
#                 "\nPlease recalculate the expected utilities more carefully. Double-check your arithmetic and make sure you're applying your utility function correctly to each outcome."
#             )
#
#     # Return best result if we got one with valid format, otherwise return last attempt
#     if best_result is not None:
#         best_result["fail_log"] = fail_log
#         best_result["failures"] = len(fail_log)
#         return best_result
#
#     ok, reasons = validate_output(last["text"], allowed_labels, expected_label, last["out_tokens"], last["stop_reason"])
#     answer_matched = "answer_not_equal_expected_label" not in reasons
#     return {
#         "text": last["text"],
#         "answer": extract_answer(last["text"]),
#         "ok": False,
#         "answer_matches_expected": answer_matched,
#         "attempts": tries,
#         "failures": tries,
#         "fail_log": fail_log,
#         "output_tokens": last["out_tokens"],
#         "stop_reason": last["stop_reason"],
#         "latency_seconds": None,
#     }


async def process_one_pair(client: anthropic.AsyncAnthropic, sem: asyncio.Semaphore, pair: ValidationPair) -> dict:
    """Process a single validation pair."""
    async with sem:
        # Generate chosen (using CARA a=0.01)
        chosen = await generate_one(
            client, pair.prompt_text, SYSTEM_PROMPT_CARA_001,
            pair.allowed_labels, pair.chosen_label, tries=MAX_TRIES
        )

        # Generate rejected (using appropriate utility function)
        if pair.rejected_type == "lin":
            system_text = SYSTEM_PROMPT_LINEAR
        else:  # too_risk
            system_text = SYSTEM_PROMPT_CARA_010

        rejected = await generate_one(
            client, pair.prompt_text, system_text,
            pair.allowed_labels, pair.rejected_label, tries=MAX_TRIES
        )

        return {
            "situation_id": pair.situation_id,
            "prompt_text": escape_newlines(sanitize_ascii_text(pair.prompt_text)),

            "chosen_expected": pair.chosen_label,
            "chosen_ok": chosen["ok"],
            "chosen_answer": chosen["answer"],
            "chosen_attempts": chosen["attempts"],
            "chosen_failures": chosen["failures"],
            "chosen_fail_log_json": json.dumps(chosen["fail_log"], ensure_ascii=True),
            "chosen_last_fail_reasons": ";".join(chosen["fail_log"][-1].get("reasons", [])) if chosen["fail_log"] else "",
            "chosen_output_tokens": chosen["output_tokens"],
            "chosen_stop_reason": chosen["stop_reason"] if chosen["stop_reason"] else "",
            "chosen_full": escape_newlines(sanitize_ascii_text(chosen["text"])),
            "chosen_answer_retries": chosen["answer_miss_count"],

            "rejected_type": pair.rejected_type,
            "rejected_expected": pair.rejected_label,
            "rejected_ok": rejected["ok"],
            "rejected_answer": rejected["answer"],
            "rejected_attempts": rejected["attempts"],
            "rejected_failures": rejected["failures"],
            "rejected_fail_log_json": json.dumps(rejected["fail_log"], ensure_ascii=True),
            "rejected_last_fail_reasons": ";".join(rejected["fail_log"][-1].get("reasons", [])) if rejected["fail_log"] else "",
            "rejected_output_tokens": rejected["output_tokens"],
            "rejected_stop_reason": rejected["stop_reason"] if rejected["stop_reason"] else "",
            "rejected_full": escape_newlines(sanitize_ascii_text(rejected["text"])),
            "rejected_answer_retries": rejected["answer_miss_count"],
        }


def atomic_save_csv(df: pd.DataFrame, path: str):
    """Atomically save DataFrame to CSV."""
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Chain-of-Thought (CoT) data for the validation set."
    )
    parser.add_argument(
        "--sample",
        nargs="?",
        const=20,
        type=int,
        metavar="N",
        help="Generate a representative sample of N pairs (default: 20 if flag used without value)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)"
    )
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Missing ANTHROPIC_API_KEY environment variable")

    client = anthropic.AsyncAnthropic(api_key=api_key)

    # Load validation pairs
    print(f"Loading validation data from {INPUT_CSV}...")
    all_pairs = load_validation_pairs(INPUT_CSV)
    print(f"Found {len(all_pairs)} valid (chosen, rejected) pairs")

    # Count by rejected type
    lin_count = sum(1 for p in all_pairs if p.rejected_type == "lin")
    too_risk_count = sum(1 for p in all_pairs if p.rejected_type == "too_risk")
    print(f"  - {lin_count} pairs with 'lin' (linear utility) rejected type")
    print(f"  - {too_risk_count} pairs with 'too_risk' (CARA a=0.10) rejected type")

    # Apply sampling if requested
    if args.sample:
        pairs = sample_representative_pairs(all_pairs, n=args.sample, seed=args.seed)
        output_csv = OUTPUT_CSV.replace(".csv", f"_sample_{args.sample}.csv")
        print(f"\n*** SAMPLE MODE: Selected {len(pairs)} representative pairs ***")
        sampled_lin = sum(1 for p in pairs if p.rejected_type == "lin")
        sampled_too_risk = sum(1 for p in pairs if p.rejected_type == "too_risk")
        print(f"  - {sampled_lin} 'lin' pairs, {sampled_too_risk} 'too_risk' pairs")
        unique_situations = len(set(p.situation_id for p in pairs))
        print(f"  - {unique_situations} unique situations")
    else:
        pairs = all_pairs
        output_csv = OUTPUT_CSV

    # Check for existing checkpoint
    done_ids = set()
    partial_df = None
    if os.path.exists(output_csv):
        try:
            partial_df = pd.read_csv(output_csv)
            # Create composite key for tracking
            done_ids = set(zip(partial_df["situation_id"], partial_df["rejected_expected"]))
            print(f"Resuming: found {len(done_ids)} completed pairs in checkpoint")
        except Exception as e:
            print(f"Warning: failed to read checkpoint: {e}")

    # Filter to remaining pairs
    remaining_pairs = [
        p for p in pairs
        if (p.situation_id, p.rejected_label) not in done_ids
    ]
    print(f"Remaining pairs to process: {len(remaining_pairs)}")

    if not remaining_pairs:
        print("All pairs already processed!")
        return

    # Process pairs
    sem = asyncio.Semaphore(CONCURRENCY)
    out_rows = []
    last_save_t = time.time()
    since_save = 0

    for i, pair in enumerate(remaining_pairs):
        result = await process_one_pair(client, sem, pair)
        out_rows.append(result)
        since_save += 1

        # Progress report
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(remaining_pairs)} pairs...")

        # Checkpoint
        now = time.time()
        if since_save >= SAVE_EVERY_ROWS or (now - last_save_t) >= SAVE_EVERY_SECONDS:
            new_df = pd.DataFrame(out_rows)
            if partial_df is not None:
                combined = pd.concat([partial_df, new_df], ignore_index=True)
            else:
                combined = new_df
            combined = combined.sort_values(["situation_id", "rejected_expected"]).reset_index(drop=True)
            atomic_save_csv(combined, output_csv)
            print(f"[checkpoint] rows={len(combined)} saved to {output_csv}")
            last_save_t = now
            since_save = 0

    # Final save
    new_df = pd.DataFrame(out_rows)
    if partial_df is not None:
        combined = pd.concat([partial_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined = combined.sort_values(["situation_id", "rejected_expected"]).reset_index(drop=True)
    atomic_save_csv(combined, output_csv)

    print(f"\nDone! Total rows: {len(combined)}")
    print(f"Saved to: {output_csv}")

    # Summary stats
    print("\nSummary:")
    print(f"  - chosen_ok (format valid + answer matches): {combined['chosen_ok'].sum()}/{len(combined)} ({100*combined['chosen_ok'].mean():.1f}%)")
    print(f"  - rejected_ok (format valid + answer matches): {combined['rejected_ok'].sum()}/{len(combined)} ({100*combined['rejected_ok'].mean():.1f}%)")

    # Answer match rates
    chosen_match = (combined['chosen_answer'] == combined['chosen_expected']).sum()
    rejected_match = (combined['rejected_answer'] == combined['rejected_expected']).sum()
    print(f"  - chosen answer match rate: {chosen_match}/{len(combined)} ({100*chosen_match/len(combined):.1f}%)")
    print(f"  - rejected answer match rate: {rejected_match}/{len(combined)} ({100*rejected_match/len(combined):.1f}%)")

    # Answer retry distribution
    if 'chosen_answer_retries' in combined.columns:
        print("\n  Answer retry distribution (chosen):")
        for retries, count in sorted(combined['chosen_answer_retries'].value_counts().items()):
            print(f"    {int(retries)} retries: {count} pairs")
        print(f"  Answer retry distribution (rejected):")
        for retries, count in sorted(combined['rejected_answer_retries'].value_counts().items()):
            print(f"    {int(retries)} retries: {count} pairs")


if __name__ == "__main__":
    asyncio.run(main())
