from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import pickle
app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model("model.h5")


# Feature names
feature_names = [
    "Mean Radius", "Mean Texture", "Mean Perimeter", "Mean Area", "Mean Compactness",
    "Mean Concavity", "Mean Concave Points", "Radius Error", "Perimeter Error", "Area Error",
    "Smoothness Error", "Concave Points Error", "Worst Radius", "Worst Texture",
    "Worst Perimeter", "Worst Area", "Worst Smoothness", "Worst Compactness",
    "Worst Concavity", "Worst Concave Points", "Worst Symmetry"
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    message = ""
    values = [0.0] * len(feature_names)

    if request.method == "POST":
        try:
            # Get user inputs
            values = [float(request.form.get(name, 0.0)) for name in feature_names]
            input_data = np.array([values])

            # Model prediction
            pred = model.predict(input_data)[0][0]

            if pred > 0.5:
                message = "🔴 The person is likely to have breast cancer."
            else:
                message = "🟢 The person is unlikely to have breast cancer."

            prediction = round(pred, 4)

        except Exception as e:
            message = f"Error: {str(e)}"

    return render_template("index.html",
                       feature_names=feature_names,
                       values=values,
                       prediction=prediction,
                       message=message,
                       zipped=zip(feature_names, values)) 
if __name__ == "__main__":
    app.run(debug=True)
