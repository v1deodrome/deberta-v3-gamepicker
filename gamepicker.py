from transformers import pipeline
from flask import Flask, render_template, request
from urllib.request import urlopen
import json
import re

# ------ necessary AI pipelines for this application ------
# https://huggingface.co/timpal0l/mdeberta-v3-base-squad2
qa_model = pipeline("question-answering", "timpal0l/mdeberta-v3-base-squad2")
# ---------------------------------------------------------

# function by c24b on StackOverflow
# https://stackoverflow.com/questions/9662346/python-code-to-remove-html-tags-from-a-string
CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
def cleanhtml(raw_html):
    cleantext = re.sub(CLEANR, '', raw_html)
    return cleantext

# summarize the description of two games, and compare them to see which one is "the most _______"
def pickGame(appid_1,appid_2,questionInput):
    print("Reading JSON data...")
    # get json data from game 1 and 2
    with urlopen(f"https://store.steampowered.com/api/appdetails?appids={appid_1}") as url:
        game1_data = json.load(url)[appid_1]['data']

    with urlopen(f"https://store.steampowered.com/api/appdetails?appids={appid_2}") as url:
        game2_data = json.load(url)[appid_2]['data']

    # ask question
    print("Querying...")
    return qa_model(
        question = f"What game sounds more {questionInput}, {game1_data['name']} or {game2_data['name']}?",
        context = "The description for {0} reads: '{1}' The description for {2} reads '{3}'".format(
            game1_data['name'],
            cleanhtml(game1_data['detailed_description']),
            game2_data['name'],
            cleanhtml(game2_data['detailed_description'])
        )
    )

# ------ FLASK STUFF BEGINS HERE ------
app = Flask(__name__)

@app.route('/', methods=['POST','GET'])
def home():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        questionInput = request.form['questionInput']
        
        # create regex for getting the appid from a store link
        regex = re.compile(r"/app/(\d+)/")
        appid_1 = regex.search(request.form['url1'])
        appid_2 = regex.search(request.form['url2'])

        # appids and questionInput do not work as parameters if they're null
        if appid_1 and appid_2 and questionInput:
            appid_1 = appid_1.group(1)
            appid_2 = appid_2.group(1)

            ai_answer = pickGame(appid_1,appid_2,questionInput)

            return render_template('index.html', answer=f"I think {ai_answer['answer']} is more {questionInput}.")
        else:
            return render_template('index.html', answer="Sorry, I think your inputs are invalid.")

if __name__ == '__main__':
    app.run()