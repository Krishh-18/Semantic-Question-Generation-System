"""
MCQ Generation Pipeline — Powered by T5 Transformer.

Uses a pre-trained question-generation T5 model (valhalla/t5-small-qg-hl)
with answer-highlight conditioning to produce high-quality, context-aware
multiple-choice questions from arbitrary input text.

Pipeline:
  1. Sentence splitting & ranking
  2. Answer candidate extraction (NER + noun chunks)
  3. T5 question generation with answer highlighting
  4. Multi-strategy distractor generation (NER-type + WordNet)
  5. MCQ assembly with quality filtering
"""

import random
import logging
import re

from transformers import T5ForConditionalGeneration, T5Tokenizer

from model.utils import (
    clean_text,
    get_sentences,
    rank_sentences,
    extract_answer_candidates,
    get_all_entities,
    generate_distractors,
    nlp,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════

# Use the HuggingFace question-generation model for best quality.
# This model is specifically fine-tuned on SQuAD for answer-aware QG.
QG_MODEL_NAME = "valhalla/t5-small-qg-hl"

_tokenizer = None
_model = None


def _load_model():
    """Lazy-load the T5 model and tokenizer (only once)."""
    global _tokenizer, _model
    if _model is None:
        logger.info("Loading question generation model: %s", QG_MODEL_NAME)
        _tokenizer = T5Tokenizer.from_pretrained(QG_MODEL_NAME)
        _model = T5ForConditionalGeneration.from_pretrained(QG_MODEL_NAME)
        _model.eval()
        logger.info("Model loaded successfully.")
    return _tokenizer, _model


# ═══════════════════════════════════════════════════════════════════════════
#  QUESTION GENERATION (T5)
# ═══════════════════════════════════════════════════════════════════════════

def _generate_question(context: str, answer: str) -> str | None:
    """
    Generate a question for a given (context, answer) pair using the T5 model.

    The valhalla/t5-small-qg-hl model expects input in the format:
      "generate question: <context with <hl> answer <hl> markers>"
    """
    tokenizer, model = _load_model()

    # Highlight the answer within the context
    # Use word-boundary aware replacement to avoid partial matches
    escaped_answer = re.escape(answer)
    highlighted = re.sub(
        rf"(?i)\b{escaped_answer}\b",
        f"<hl> {answer} <hl>",
        context,
        count=1,
    )

    # If regex didn't match (e.g., answer spans multiple words with different casing),
    # fall back to simple string replacement
    if "<hl>" not in highlighted:
        idx = context.lower().find(answer.lower())
        if idx == -1:
            return None
        original_span = context[idx : idx + len(answer)]
        highlighted = context[:idx] + f"<hl> {original_span} <hl>" + context[idx + len(answer):]

    input_text = f"generate question: {highlighted}"

    # Tokenize and generate
    inputs = tokenizer(
        input_text,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=72,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )

    question = tokenizer.decode(outputs[0], skip_special_tokens=True)
    question = question.strip()

    # Basic quality check
    if not question or len(question) < 8:
        return None
    if not question.endswith("?"):
        question += "?"

    return question


# ═══════════════════════════════════════════════════════════════════════════
#  MCQ ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════

def generate_mcqs(text: str, num: int = 5) -> list[dict]:
    """
    Generate `num` multiple-choice questions from the input text.

    Returns a list of dicts:
    [
      {
        "question": "What is ...?",
        "options":  ["A", "B", "C", "D"],
        "answer":   "A",
        "context":  "Source sentence..."
      },
      ...
    ]
    """
    text = clean_text(text)

    if len(text) < 50:
        raise ValueError("Input text is too short. Please provide at least a few sentences.")

    # ── Step 1: Split and rank sentences ──
    sentences = get_sentences(text, min_length=30)
    if not sentences:
        raise ValueError("Could not extract meaningful sentences from the input.")

    ranked = rank_sentences(sentences, top_n=num * 3)

    # ── Step 2: Extract all entities from full text (for distractors) ──
    all_entities = get_all_entities(text)

    # ── Step 3: Generate MCQs ──
    mcqs = []
    used_answers = set()   # Avoid duplicate answer-based questions
    used_questions = set()  # Avoid duplicate questions

    for sentence in ranked:
        if len(mcqs) >= num:
            break

        # Extract answer candidates from this sentence
        candidates = extract_answer_candidates(sentence, all_entities)

        if not candidates:
            continue

        # Try each candidate until we get a valid question
        for candidate in candidates:
            if len(mcqs) >= num:
                break

            answer_text = candidate["text"]
            answer_label = candidate["label"]

            # Skip if we've already used this answer
            if answer_text.lower() in used_answers:
                continue

            # Verify the answer actually appears in the sentence
            if answer_text.lower() not in sentence.lower():
                continue

            # ── Generate question using T5 ──
            question = _generate_question(sentence, answer_text)
            if question is None:
                continue

            # Skip duplicate or near-duplicate questions
            q_key = question.lower().strip("? ")
            if q_key in used_questions:
                continue

            # ── Generate distractors ──
            distractors = generate_distractors(
                answer=answer_text,
                answer_label=answer_label,
                all_entities=all_entities,
                sentence=sentence,
                needed=3,
            )

            # Need at least 2 distractors for a viable MCQ
            if len(distractors) < 2:
                continue

            # Pad to 3 if we only got 2
            while len(distractors) < 3:
                distractors.append(f"None of the above")

            # ── Assemble the MCQ ──
            options = [answer_text] + distractors[:3]
            random.shuffle(options)

            mcqs.append({
                "question": question,
                "options":  options,
                "answer":   answer_text,
                "context":  sentence,
            })

            used_answers.add(answer_text.lower())
            used_questions.add(q_key)
            break  # One question per sentence for diversity

    if not mcqs:
        raise ValueError(
            "Could not generate any questions from the provided text. "
            "Try providing a longer, more detailed passage."
        )

    return mcqs[:num]