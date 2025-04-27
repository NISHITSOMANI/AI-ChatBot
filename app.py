from flask import Flask, render_template, request, jsonify
from utils import QnA  # Import the optimized QnA class
from threading import Thread
app = Flask(__name__)

# Initialize the QnA class (loads QnA data, vectorizer, etc.)
qna = QnA()

# Track learning state
learning_state = {"learning": False, "pending_question": None}

# Background task to save new QnA data without blocking the user
def save_new_qna_async(new_qna):
    qna.save_qna(new_qna)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    global learning_state
    user_input = request.form["msg"]
    if user_input == "no":
        learning_state["learning"] = False
        learning_state["pending_question"] = None
        return jsonify({"answer": "Okay, I won't learn that. ❌"})
    # Check if bot is waiting for a new answer (learning mode)
    if learning_state["learning"]:
        # Save new QnA asynchronously to avoid blocking
        new_qna = {
            "question": learning_state["pending_question"],
            "answer": user_input
        }
        thread = Thread(target=save_new_qna_async, args=(new_qna,))
        thread.start()

        learning_state["learning"] = False
        learning_state["pending_question"] = None
        return jsonify({"answer": "Thanks! I've learned a new answer. ✅"})

    # Check if the input is a greeting
    if qna.is_greeting(user_input):
        return jsonify({"answer": "Hello! 👋 I am Vaultify Bot, your secure assistant. How can I help you today?"})

    # Try to find an answer
    matched_answer = qna.find_answer(user_input)
    if matched_answer:
        return jsonify({"answer": matched_answer})
    else:
        # If no answer is found, switch to learning mode
        learning_state["learning"] = True
        learning_state["pending_question"] = user_input
        return jsonify({"answer": "I don't know the answer yet. Would you like to teach me? (please reply with the answer)"})

if __name__ == "__main__":
    app.run(debug=False)
