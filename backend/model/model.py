import random
import spacy

# load NLP model
nlp = spacy.load("en_core_web_sm")


# -----------------------------
# Extract definition pairs
# -----------------------------
def extract_definitions(text):
    doc = nlp(text)
    pairs = []

    for sent in doc.sents:
        sentence = sent.text.strip()

        if " is " in sentence:
            parts = sentence.split(" is ")

            subject = parts[0].strip()
            definition = parts[1].strip()

            # clean subject (remove stopwords noise)
            if len(subject.split()) <= 6:
                pairs.append((subject, definition))

    return pairs


# -----------------------------
# Extract named entities
# -----------------------------
def extract_entities(text):
    doc = nlp(text)
    entities = list(set([ent.text for ent in doc.ents]))
    return entities


# -----------------------------
# Generate MCQs
# -----------------------------
def generate_mcqs(text, num=5):
    definitions = extract_definitions(text)
    entities = extract_entities(text)

    mcqs = []

    for subject, definition in definitions:
        if len(mcqs) >= num:
            break

        question = f"What is {subject}?"
        correct = definition.capitalize()

        # distractors from entities
        distractors = []
        for ent in entities:
            if ent.lower() not in correct.lower() and len(distractors) < 3:
                distractors.append(ent)

        fallback = [
            "A natural process",
            "A scientific method",
            "A theoretical concept"
        ]

        while len(distractors) < 3:
            distractors.append(fallback[len(distractors)])

        options = [correct] + distractors[:3]
        random.shuffle(options)

        mcqs.append({
            "question": question,
            "options": options,
            "answer": correct
        })

    return mcqs[:num]