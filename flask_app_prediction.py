from flask import Flask
import pickle
app = Flask(__name__)

model = pickle.load(open('modelLR.pkl', 'rb'))


@app.route('/')
def app_running():
    scores = [[0,0,0,0,0,0,0,0,0,0]]
    output = model.predict(scores)[0]
    output = f'The prediction for the score is {output}'
    return str(output)



if __name__ == '__main__':
    app.run(port=9000, debug = True)

