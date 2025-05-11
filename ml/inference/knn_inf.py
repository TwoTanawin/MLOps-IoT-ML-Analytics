import joblib
import numpy as np
import time
from scipy.signal import butter, filtfilt
import pandas as pd
from collections import Counter

def apply_lowpass_filter(df: pd.DataFrame, cutoff: float = 0.1, order: int = 2) -> pd.DataFrame:
    b, a = butter(order, cutoff, btype='low', analog=False)
    
    filtered_df = df.copy()
    for column in df.columns:
        data = df[column].values
        if len(data) < 10:
            raise ValueError(f"Not enough data to filter column '{column}'. Need at least 10 points.")
        filtered_data = filtfilt(b, a, data)
        filtered_df[f"{column}_filtered"] = filtered_data
    
    return filtered_df

def main():
    model = joblib.load(r'G:\AIT\MLOps\ProjectAnalytics\ml\model/gb_model.joblib')

    data = [
        [0.1, 0.33, 0.34, 10000.25],
        [0.1, 0.33, 111.34, 23.25],
        [32.1, 43.3, 111.34, 23.25],
        [33.1, 44.3, 110.34, 24.25],
        [34.1, 45.3, 109.34, 25.25],
        [35.1, 46.3, 108.34, 26.25],
        [36.1, 47.3, 107.34, 27.25],
        [37.1, 48.3, 106.34, 28.25],
        [38.1, 49.3, 105.34, 29.25],
        [39.1, 50.3, 104.34, 30.25],
    ]
    columns = ["ph_value", "do_value", "temp_value", "salinity_value"]
    df = pd.DataFrame(data, columns=columns)
    
    df_filtered = apply_lowpass_filter(df)

    # Extract filtered features
    feature_cols = ["ph_value_filtered", "do_value_filtered", "temp_value_filtered", "salinity_value_filtered"]
    X = df_filtered[feature_cols]

    # Class names
    class_names = ['Clean', 'Low pH', 'High pH', 'Chemical', 'Salt', 'Organic']

    # Predictions & probabilities
    pred_classes = model.predict(X)
    pred_probs = model.predict_proba(X)
    
    for i, (cls, probs) in enumerate(zip(pred_classes, pred_probs)):
        confidence = probs[cls]
        print(f"Row {i}: Predicted = {class_names[cls]} (Index {cls}), Confidence = {confidence:.4f}")

    # Find the row with highest confidence prediction
    row_idx, class_idx = np.unravel_index(np.argmax(pred_probs), pred_probs.shape)
    max_confidence = pred_probs[row_idx, class_idx]

    print(f"\nMost confident prediction at row {row_idx}:")
    print(f"Predicted class index: {class_idx}")
    print(f"Predicted class name: {class_names[class_idx]}")
    print(f"Confidence score: {max_confidence:.4f}")

if __name__=="__main__":
    start = time.time()
    
    main()

    end = time.time()

    print(f"Time taken: {end - start:.4f} seconds")