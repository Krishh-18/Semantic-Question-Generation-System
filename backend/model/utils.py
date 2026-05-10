"""
Utility functions for the MCQ generation pipeline.
Handles answer extraction, distractor generation, and text processing.
"""

import random
import re
import string
from collections import Counter

import nltk
import spacy

# ── Ensure required NLTK data is available ──────────────────────────────────
for resource in ["wordnet", "stopwords", "averaged_perceptron_tagger_eng", "punkt_tab"]:
    try:
        nltk.data.find(f"corpora/{resource}" if resource in ("wordnet", "stopwords") else f"taggers/{resource}" if "tagger" in resource else f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))

# ── Load spaCy model ────────────────────────────────────────────────────────
nlp = spacy.load("en_core_web_sm")


# ═══════════════════════════════════════════════════════════════════════════
#  TEXT PROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """Normalize whitespace and strip artifacts from input text."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_sentences(text: str, min_length: int = 30) -> list[str]:
    """
    Split text into sentences, keeping only those that are long enough
    to contain meaningful content for question generation.
    """
    doc = nlp(text)
    sentences = []
    for sent in doc.sents:
        s = sent.text.strip()
        if len(s) >= min_length and any(c.isalpha() for c in s):
            sentences.append(s)
    return sentences


def rank_sentences(sentences: list[str], top_n: int = 10) -> list[str]:
    """
    Rank sentences by information density using a simple keyword-frequency
    heuristic.  Sentences with more unique, non-stop-word tokens score higher.
    """
    scored = []
    for sent in sentences:
        tokens = [t.lower() for t in sent.split() if t.lower() not in STOP_WORDS and t.isalpha()]
        unique_ratio = len(set(tokens)) / max(len(tokens), 1)
        scored.append((sent, len(tokens) * unique_ratio))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scored[:top_n]]


# ═══════════════════════════════════════════════════════════════════════════
#  ANSWER EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

# NER label groups for type-aware distractor matching
ENTITY_TYPE_GROUPS = {
    "PERSON":  {"PERSON"},
    "ORG":     {"ORG"},
    "GPE":     {"GPE", "LOC"},
    "LOC":     {"GPE", "LOC"},
    "DATE":    {"DATE", "TIME"},
    "TIME":    {"DATE", "TIME"},
    "MONEY":   {"MONEY", "QUANTITY", "CARDINAL"},
    "CARDINAL": {"CARDINAL", "QUANTITY", "MONEY"},
    "PERCENT": {"PERCENT", "CARDINAL"},
    "QUANTITY": {"QUANTITY", "CARDINAL", "MONEY"},
}


def extract_answer_candidates(sentence: str, full_doc_entities: list[dict]) -> list[dict]:
    """
    Extract potential answer spans from a sentence.
    Returns a list of dicts: {"text": ..., "label": ..., "start": ..., "end": ...}

    Priority:
      1. Named entities (PERSON, ORG, GPE, DATE, etc.)
      2. Noun chunks that are meaningful (not pronouns, not stop words)
    """
    doc = nlp(sentence)
    candidates = []
    seen_texts = set()

    # ── Named entities first (highest quality answers) ──
    for ent in doc.ents:
        text = ent.text.strip()
        if text.lower() in seen_texts or len(text) < 2:
            continue
        if ent.label_ in ("CARDINAL", "ORDINAL") and not any(c.isdigit() for c in text):
            continue
        seen_texts.add(text.lower())
        candidates.append({
            "text":  text,
            "label": ent.label_,
            "start": ent.start_char,
            "end":   ent.end_char,
        })

    # ── Noun chunks as fallback ──
    for chunk in doc.noun_chunks:
        text = chunk.text.strip()
        # Skip pronouns, determiners-only, stop words
        if text.lower() in seen_texts:
            continue
        if chunk.root.pos_ in ("PRON", "DET"):
            continue
        if text.lower() in STOP_WORDS or len(text) < 3:
            continue
        # Skip chunks that are too long (likely full clauses)
        if len(text.split()) > 5:
            continue
        seen_texts.add(text.lower())
        candidates.append({
            "text":  text,
            "label": "NOUN_CHUNK",
            "start": chunk.start_char,
            "end":   chunk.end_char,
        })

    return candidates


def get_all_entities(text: str) -> list[dict]:
    """
    Extract all named entities from the full input text.
    Returns list of {"text": ..., "label": ...}.
    """
    doc = nlp(text)
    entities = []
    seen = set()
    for ent in doc.ents:
        key = (ent.text.strip().lower(), ent.label_)
        if key not in seen and len(ent.text.strip()) >= 2:
            seen.add(key)
            entities.append({"text": ent.text.strip(), "label": ent.label_})
    return entities


# ═══════════════════════════════════════════════════════════════════════════
#  DISTRACTOR GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation for comparison."""
    return text.lower().strip().strip(string.punctuation)


def _is_too_similar(a: str, b: str) -> bool:
    """Check if two strings are too similar to be a useful distractor pair."""
    na, nb = _normalize(a), _normalize(b)
    if na == nb:
        return True
    if na in nb or nb in na:
        return True
    # Check word overlap
    words_a = set(na.split())
    words_b = set(nb.split())
    if not words_a or not words_b:
        return True
    overlap = len(words_a & words_b) / min(len(words_a), len(words_b))
    return overlap > 0.8


def generate_distractors_wordnet(answer: str, count: int = 6) -> list[str]:
    """
    Use WordNet to find semantically related but distinct alternatives
    to the answer.  Looks for:
      - Hypernyms  (broader terms)
      - Hyponyms   (narrower terms)
      - Co-hyponyms (siblings under the same hypernym)
    """
    distractors = set()

    # Tokenize the answer and look up each meaningful word
    words = [w for w in answer.split() if w.lower() not in STOP_WORDS and w.isalpha()]
    if not words:
        return []

    for word in words:
        synsets = wn.synsets(word)
        for syn in synsets[:3]:  # limit breadth
            # Co-hyponyms (siblings)
            for hypernym in syn.hypernyms():
                for hypo in hypernym.hyponyms():
                    for lemma in hypo.lemmas():
                        name = lemma.name().replace("_", " ")
                        if not _is_too_similar(name, answer):
                            distractors.add(name)

            # Direct hyponyms
            for hypo in syn.hyponyms():
                for lemma in hypo.lemmas():
                    name = lemma.name().replace("_", " ")
                    if not _is_too_similar(name, answer):
                        distractors.add(name)

            # Direct hypernyms
            for hyper in syn.hypernyms():
                for lemma in hyper.lemmas():
                    name = lemma.name().replace("_", " ")
                    if not _is_too_similar(name, answer):
                        distractors.add(name)

    result = list(distractors)
    random.shuffle(result)
    return result[:count]


def generate_distractors_ner(
    answer: str,
    answer_label: str,
    all_entities: list[dict],
    count: int = 6,
) -> list[str]:
    """
    Find distractors by matching the NER type of the correct answer.
    E.g., if the answer is a PERSON, pull other PERSON entities from the text.
    """
    compatible_labels = ENTITY_TYPE_GROUPS.get(answer_label, {answer_label})
    distractors = []

    for ent in all_entities:
        if ent["label"] in compatible_labels and not _is_too_similar(ent["text"], answer):
            distractors.append(ent["text"])

    random.shuffle(distractors)
    return distractors[:count]


def generate_distractors(
    answer: str,
    answer_label: str,
    all_entities: list[dict],
    sentence: str,
    needed: int = 3,
) -> list[str]:
    """
    Master distractor generation function.  Combines multiple strategies
    and filters the result to exactly `needed` high-quality distractors.

    Strategy priority:
      1. NER-type matching   (same entity type from the passage)
      2. WordNet co-hyponyms (semantically related terms)
      3. Noun-chunk fallback (other noun chunks from the sentence)
    """
    pool: list[str] = []

    # ── Strategy 1: NER-type matching ──
    ner_distractors = generate_distractors_ner(answer, answer_label, all_entities)
    pool.extend(ner_distractors)

    # ── Strategy 2: WordNet ──
    if len(pool) < needed * 2:  # Only go to WordNet if NER didn't give enough
        wn_distractors = generate_distractors_wordnet(answer, count=needed * 2)
        pool.extend(wn_distractors)

    # ── Strategy 3: Noun chunk fallback ──
    if len(pool) < needed:
        doc = nlp(sentence)
        for chunk in doc.noun_chunks:
            text = chunk.text.strip()
            if not _is_too_similar(text, answer) and len(text) > 2:
                pool.append(text)

    # ── Filter and deduplicate ──
    filtered = filter_distractors(pool, answer, needed)
    return filtered


def filter_distractors(
    candidates: list[str],
    correct_answer: str,
    needed: int = 3,
) -> list[str]:
    """
    Filter a pool of distractor candidates:
      - Remove duplicates (case-insensitive)
      - Remove candidates too similar to the correct answer
      - Remove very short or empty candidates
      - Return exactly `needed` distractors
    """
    seen = set()
    result = []

    for d in candidates:
        d = d.strip()
        if not d or len(d) < 2:
            continue
        key = _normalize(d)
        if key in seen:
            continue
        if _is_too_similar(d, correct_answer):
            continue
        seen.add(key)
        result.append(d)

        if len(result) >= needed:
            break

    return result
