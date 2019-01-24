from flask import Flask
import pickle


app = Flask(__name__)

model = pickle.load(open('DataScienceModel.pkl', 'rb'))

@app.route('/')
def app_running():
    scores = [[ 0.00740741, 0.63518519, 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 1.        , 0.21272999,
       1.41745251]]
    output = model.predict(scores)[0]
    output = f'The prediction for these set out inputs gives us {output}'
    return str(output)

if __name__ == '__main__':
    app.run(port = 9000, debug = True)