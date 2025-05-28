from flask import Flask, request, jsonify
import pandas as pd

from model_utils import predict_new_cases

app = Flask(__name__)



@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        df = pd.DataFrame(data)
        if 'full_report' not in df.columns:
            return jsonify({"error": "Required fields: 'full_report'"}), 400

        predictions = predict_new_cases(df)
        return jsonify(predictions)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
