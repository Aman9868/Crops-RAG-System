from flask import Flask, request, jsonify, render_template
from conv import answer_question

app = Flask(__name__)

# Serve index.html at the root URL
@app.route('/')
def index():
    return render_template('sample.html')

# Define route for answering questions
@app.route('/answer_question', methods=['POST'])
def handle_question():
    data = request.json
    question = data['question']
    chat_history = data.get('chat_history', [])
    
    # Get the answer and updated chat history
    answer, updated_chat_history = answer_question(question, chat_history)
    
    return jsonify({"answer": answer, "chat_history": updated_chat_history})

if __name__ == '__main__':
    app.run(debug=True)
