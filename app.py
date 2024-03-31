from functions import answer_question
from conv import answer_questionss
from flask import Flask, render_template, request,jsonify
app = Flask(__name__)
@app.route('/')
def home():
    return render_template("index.html")

def json_to_chathistory(history):
    if history is None:
        return []
    
    res = []
    for each in history:
        res.append((each[0], each[1]))
    return res

@app.route("/chat", methods=["POST"])
def chat():
    body = request.get_json()
    # print(body.get("input"), body.get("chat_history"))
    query = body.get("input")
    chat_history = json_to_chathistory(body.get("chat_history"))

    res = answer_questionss(query, chat_history)

    print(res)
    
    return jsonify(
        {
            "query": res["question"],
            "result": res["answer"],
            "chat_history": res["chat_history"] + [(res["question"], res["answer"])],
        }
    )
if __name__ == '__main__':
    app.run(debug=True)