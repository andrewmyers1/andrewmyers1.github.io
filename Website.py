from flask import Flask, render_template, jsonify, request
import WebsiteFunctions
import requests
import json

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/get_time', methods=['GET'])
def get_time():
    time = WebsiteFunctions.get_time()
    return jsonify(time)


@app.route('/get_response', methods=['GET'])
def get_response():
    prompt = request.args.get('prompt', '')
    message = WebsiteFunctions.get_request(prompt)
    return jsonify(message)


@app.route('/get_normal_gpt', methods=['GET'])
def get_normal_gpt():
    regular_prompt = request.args.get('regular_prompt', '')
    message = WebsiteFunctions.normal_gpt(regular_prompt)
    return jsonify(message)


@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'fileInput' not in request.files:
        return 'No file part'

    file = request.files['fileInput']

    if ".txt" not in file.filename:
        return 'Must select a .txt file'

    path = r"\\loki\Files\AMyers\AI\EmbeddingTrainingFiles\\" + str(file.filename)

    WebsiteFunctions.add_embeddings(path)

    return 'File successfully added to embeddings.csv'


@app.route('/api/<string:ticketID>/<string:prompt>', methods=['GET'])
def api(ticketID, prompt):
    data = WebsiteFunctions.get_request(prompt)

    url = f"https://uluro.zendesk.com/api/v2/tickets/{ticketID}"

    print("url: " + url)
    print("question: " + prompt)

    payload = json.loads("""{
      "ticket": {
        "comment": {
          "body": "ERROR",
          "public": false
        }
      }
    }""")

    payload["ticket"]["comment"]["body"] = data

    print("Payload: " + str(payload))

    headers = {
        "Content-Type": "application/json",
    }

    response = requests.request(
        "PUT",
        url,
        auth=('bill@transfrm.com', 'billt'),
        headers=headers,
        json=payload
    )

    print("Response: " + str(response))

    return jsonify(data)




if __name__ == '__main__':
    app.run(debug=True)
