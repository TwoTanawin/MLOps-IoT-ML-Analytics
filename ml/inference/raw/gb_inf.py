import joblib
import numpy as np
import time


def main():
    # Load model
    model = joblib.load(r'G:\AIT\MLOps\ProjectAnalytics\ml\model\raw\raw_gb_model.joblib')

    # Sample input
    sample = np.array([8.982, 2.643, 29.558, 414.272])

    # Reshape for model
    filtered_input = np.array(sample).reshape(1, -1)

    # Predict
    prediction = model.predict(filtered_input)
    prediction_proba = model.predict_proba(filtered_input)

    # Optional: map to class name
    class_names = ['Clean', 'Low pH', 'High pH', 'Chemical', 'Salt', 'Organic']
    predicted_index = prediction[0]
    confidence = prediction_proba[0][predicted_index]

    print("Predicted class index:", predicted_index)
    print("Predicted class name:", class_names[predicted_index])
    print("Confidence (probability):", round(confidence * 100, 2), "%")

if __name__=="__main__":
    start = time.time()
    
    main()

    end = time.time()

    print(f"Time taken: {end - start:.4f} seconds")