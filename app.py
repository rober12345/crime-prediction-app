from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model
with open("crime_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        year = int(request.form["year"])
        month = int(request.form["month"])
        prediction = model.predict(np.array([[year, month]]))[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)

