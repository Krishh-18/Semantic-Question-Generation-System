from flask import Flask, request, jsonify
from flask_cors import CORS
from model.model import generate_mcqs

app = Flask(__name__)
CORS(app)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    text = data["text"]

    mcqs = generate_mcqs(text, 5)

    return jsonify({
        "mcqs": mcqs
    })

if __name__ == "__main__":
    app.run(debug=True)