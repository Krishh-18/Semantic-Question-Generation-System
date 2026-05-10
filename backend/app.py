"""
Flask API for the Semantic MCQ Generation System.
"""

import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from model.model import generate_mcqs

# ── Logging ──
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)


# ═══════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


@app.route("/generate", methods=["POST"])
def generate():
    """
    Generate MCQs from input text.

    Request JSON:
      {
        "text": "...",
        "num_questions": 5   (optional, default 5, max 10)
      }

    Response JSON:
      {
        "mcqs": [ { "question": ..., "options": [...], "answer": ..., "context": ... }, ... ]
      }
    """
    data = request.get_json(silent=True)

    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request body."}), 400

    text = data["text"].strip()
    num_questions = min(int(data.get("num_questions", 5)), 10)

    if len(text) < 50:
        return jsonify({"error": "Text is too short. Please provide at least a few sentences."}), 400

    if len(text) > 15000:
        return jsonify({"error": "Text is too long. Please limit to 15,000 characters."}), 400

    try:
        logger.info("Generating %d MCQs from %d chars of text...", num_questions, len(text))
        mcqs = generate_mcqs(text, num=num_questions)
        logger.info("Successfully generated %d MCQs.", len(mcqs))
        return jsonify({"mcqs": mcqs})

    except ValueError as e:
        logger.warning("ValueError: %s", e)
        return jsonify({"error": str(e)}), 422

    except Exception as e:
        logger.exception("Unexpected error during MCQ generation.")
        return jsonify({"error": "An unexpected error occurred. Please try again."}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)