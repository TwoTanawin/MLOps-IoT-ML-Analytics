import joblib
import numpy as np
import time
from scipy.signal import butter, filtfilt
import pandas as pd
from collections import Counter


def main():
    # Load model
    model = joblib.load(r'G:\AIT\MLOps\ProjectAnalytics\ml\model\gb_model.joblib')

    # Sample input
    sample = np.array([0.1, 0.33, 0.34, 10000.25])


    # Reshape for model
    filtered_input = np.array(sample).reshape(1, -1)

    # Predict
    prediction = model.predict(filtered_input)
    print("Predicted class index:", prediction[0])

    # Optional: map to class name
    class_names = ['Clean', 'Low pH', 'High pH', 'Chemical', 'Salt', 'Organic']
    print("Predicted class name:", class_names[prediction[0]])

if __name__=="__main__":
    start = time.time()
    
    main()

    end = time.time()

    print(f"Time taken: {end - start:.4f} seconds")