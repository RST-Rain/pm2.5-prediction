import torch
import numpy as np
import joblib
from model import AirPollutionModel

# Function for inference
def predict_air_pollution(model_path, scaler_path, new_data):
    """
    Predicts PM2.5 for new input data using the trained model and scaler.
    
    Args:
        model_path: Path to the saved model (.pth).
        scaler_path: Path to the saved scaler (.pkl).
        new_data: Dictionary with features, e.g., 
                  {'Temperature': 17.0, 'Precipitation': 0, 'RH': 70, 'WS': 2.5, 
                   'WD': 90, 'SunShine': 0.5, 'GloblRad': 4.0}
    
    Returns:
        Predicted PM2.5 value.
    """
    # Load scaler
    scaler = joblib.load(scaler_path)
    
    # Load model
    input_dim = 7  # Number of features: Temperature, Precipitation, RH, WS, WD, SunShine, GloblRad
    model = AirPollutionModel(input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Convert new_data to numpy array in correct order
    features_order = ['Temperature', 'Precipitation', 'RH', 'WS', 'WD', 'SunShine', 'GloblRad']
    if isinstance(new_data, dict):
        new_array = np.array([[new_data[feat] for feat in features_order]])
    else:
        new_array = np.array([new_data])
    
    # Scale the new data
    new_array_scaled = scaler.transform(new_array)
    
    # Predict
    with torch.no_grad():
        input_tensor = torch.tensor(new_array_scaled, dtype=torch.float32)
        prediction = model(input_tensor).squeeze().item()
    
    return prediction

# Example usage
if __name__ == "__main__":
    model_path = './model.pth'
    scaler_path = './scaler.pkl'
    new_data = {
        'Temperature': 17.0,
        'Precipitation': 0,
        'RH': 70,
        'WS': 2.5,
        'WD': 90,
        'SunShine': 0.5,
        'GloblRad': 4.0
    }
    pred = predict_air_pollution(model_path, scaler_path, new_data)
    print(f"Predicted PM2.5: {pred}")