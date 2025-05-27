from flask import Flask, request, render_template, redirect, url_for
import pandas as pd

from .model_utils import predict_new_cases

app = Flask(__name__)



@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["csv_file"]
        if file:
            df = pd.read_csv(file)
            predictions = predict_new_cases(df)
            return render_template("index.html", tables=[predictions.to_html(classes='data')], titles=predictions.columns.values)
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
