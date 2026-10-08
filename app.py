from flask import Flask, render_template, request, jsonify
import gensim.downloader as api
import re

app = Flask(__name__)

print("Loading GloVe embeddings...")
glove = api.load("glove-wiki-gigaword-50")
print("GloVe loaded successfully.")


def calculate_words(expression):
    tokens = re.findall(r"[A-Za-z]+|[+-]", expression.lower())

    if not tokens:
        raise ValueError("Please enter a word calculation.")

    positive = []
    negative = []
    current_sign = "+"

    for token in tokens:
        if token in ["+", "-"]:
            current_sign = token
        else:
            if token not in glove:
                raise ValueError(f'"{token}" is not in the vocabulary.')

            if current_sign == "+":
                positive.append(token)
            else:
                negative.append(token)

    if not positive:
        raise ValueError("Please enter at least one positive word.")

    results = glove.most_similar(
        positive=positive,
        negative=negative,
        topn=5
    )

    return results


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/calculate", methods=["POST"])
def calculate():
    try:
        data = request.get_json()
        expression = data.get("expression", "").strip()

        if not expression:
            return jsonify({"error": "Please enter a word calculation."}), 400

        results = calculate_words(expression)

        return jsonify({
            "expression": expression,
            "results": [
                {
                    "word": word,
                    "score": round(float(score), 4)
                }
                for word, score in results
            ]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
