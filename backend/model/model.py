import random
import spacy

nlp = spacy.load("en_core_web_sm")


# -----------------------------
# Clean sentence list
# -----------------------------
def get_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]


# -----------------------------
# Extract entities
# -----------------------------
def extract_entities(doc):
    return list(set([ent.text for ent in doc.ents]))


# -----------------------------
# Generate MCQs (IMPROVED)
# -----------------------------
def generate_mcqs(text, num=5):
    doc = nlp(text)
    sentences = get_sentences(text)
    entities = extract_entities(doc)

    mcqs = []

    for sent in sentences:
        if len(mcqs) >= num:
            break

        sentence = sent.strip()

        # ------------------ TYPE 1: Definition ------------------
        if " is " in sentence:
            parts = sentence.split(" is ")
            subject = parts[0].strip()
            definition = parts[1].strip()

            if len(subject.split()) <= 6:
                question = f"What is {subject}?"
                correct = definition.capitalize()

        # ------------------ TYPE 2: Usage ------------------
        elif " used " in sentence:
            words = sentence.split()
            subject = words[0]

            question = f"What is {subject} used for?"
            correct = sentence.capitalize()

        # ------------------ TYPE 3: Location ------------------
        elif " occurs " in sentence or " occurs in " in sentence:
            words = sentence.split()
            subject = words[0]

            question = f"Where does {subject} occur?"
            correct = sentence.capitalize()

        # ------------------ TYPE 4: General ------------------
        else:
            # skip weak sentences
            continue

        # ------------------ DISTRACTORS ------------------
        distractors = []

        for ent in entities:
            if ent.lower() not in correct.lower() and len(distractors) < 3:
                distractors.append(ent)

        fallback = [
            "A scientific concept",
            "A natural process",
            "A technical system"
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