import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import joblib
from model import AirQualityDataset, AirPollutionModel

# Function to calculate accuracy for regression
def calculate_accuracy(predictions, targets, threshold=5.0):
    """
    Calculates accuracy as the proportion of predictions within a threshold of the actual values.
    Args:
        predictions: Predicted values (tensor or numpy array).
        targets: Actual values (tensor or numpy array).
        threshold: Maximum allowed absolute error for a prediction to be considered correct (default: 5 μg/m³).
    Returns:
        Accuracy as a float (proportion of correct predictions).
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()  # Detach before converting to numpy
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()  # Detach before converting to numpy
    errors = np.abs(predictions - targets)
    correct = np.sum(errors < threshold)
    return correct / len(targets)

# Function to load and preprocess the data
def load_and_preprocess_data(air_quality_path, rainfall_path, weather_path):
    """
    Loads and merges the three CSV files, handles missing values, and prepares features and target.
    Returns: X (features), y (target), scaler (StandardScaler).
    """
    air_df = pd.read_csv(air_quality_path)
    rain_df = pd.read_csv(rainfall_path)
    weather_df = pd.read_csv(weather_path)
    
    air_df.rename(columns={'date': 'Date'}, inplace=True)
    weather_df.rename(columns={'timestamp': 'Date'}, inplace=True)
    
    air_df['Date'] = pd.to_datetime(air_df['Date'], format='%Y/%m/%d')
    rain_df['Date'] = pd.to_datetime(rain_df['Date'], format='%Y/%m/%d')
    weather_df['Date'] = pd.to_datetime(weather_df['Date'], format='%Y/%m/%d')
    
    merged_df = pd.merge(air_df, rain_df, on='Date', how='inner')
    merged_df = pd.merge(merged_df, weather_df, on='Date', how='inner')
    
    features = ['Temperature', 'Precipitation', 'RH', 'WS', 'WD', 'SunShine', 'GloblRad']
    target = 'pm2.5'
    
    merged_df.replace('', np.nan, inplace=True)
    merged_df.fillna(0, inplace=True)
    
    X = merged_df[features].values
    y = merged_df[target].values
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y, scaler

# Function to train the model and save losses and accuracies
def train_model(air_quality_path, rainfall_path, weather_path, epochs=100, batch_size=2, lr=0.001, threshold=5.0):
    """
    Trains the MLP model, tracks training and validation losses and accuracies, saves them to CSV,
    and plots loss and accuracy curves. Saves the model and scaler.
    Returns: model, scaler, train_losses, test_losses, train_accuracies, test_accuracies.
    """
    X, y, scaler = load_and_preprocess_data(air_quality_path, rainfall_path, weather_path)
    
    # Split into train/test (validation set is called 'test' here)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create datasets and dataloaders
    train_dataset = AirQualityDataset(X_train, y_train)
    test_dataset = AirQualityDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model, loss, optimizer
    input_dim = X.shape[1]
    model = AirPollutionModel(input_dim)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Lists to store losses and accuracies
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_train_loss = 0.0
        epoch_train_correct = 0
        total_train_samples = 0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            
            # Accumulate loss and accuracy
            epoch_train_loss += loss.item() * inputs.size(0)
            epoch_train_correct += calculate_accuracy(outputs.squeeze(), targets, threshold) * inputs.size(0)
            total_train_samples += inputs.size(0)
        
        # Average training loss and accuracy for the epoch
        epoch_train_loss /= len(train_dataset)
        epoch_train_accuracy = epoch_train_correct / total_train_samples
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_accuracy)
        
        # Calculate validation loss and accuracy
        model.eval()
        epoch_test_loss = 0.0
        epoch_test_correct = 0
        total_test_samples = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                epoch_test_loss += loss.item() * inputs.size(0)
                epoch_test_correct += calculate_accuracy(outputs.squeeze(), targets, threshold) * inputs.size(0)
                total_test_samples += inputs.size(0)
        epoch_test_loss /= len(test_dataset)
        epoch_test_accuracy = epoch_test_correct / total_test_samples
        test_losses.append(epoch_test_loss)
        test_accuracies.append(epoch_test_accuracy)
        
        model.train()  # Switch back to training mode
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_train_loss:.4f}, "
                  f"Test Loss: {epoch_test_loss:.4f}, Train Acc: {epoch_train_accuracy:.4f}, "
                  f"Test Acc: {epoch_test_accuracy:.4f}")
    
    # Save losses and accuracies to CSV
    metrics_df = pd.DataFrame({
        'Epoch': range(1, epochs + 1),
        'Train_Loss': train_losses,
        'Test_Loss': test_losses,
        'Train_Accuracy': train_accuracies,
        'Test_Accuracy': test_accuracies
    })
    metrics_df.to_csv('./runs/metrics.csv', index=False)
    
    # Plot loss and accuracy curves
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', color='blue')
    plt.plot(range(1, epochs + 1), test_losses, label='Test Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error (MSE) Loss')
    plt.title('Training and Test Loss Curve')
    plt.legend()
    plt.grid(True)
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy', color='blue')
    plt.plot(range(1, epochs + 1), test_accuracies, label='Test Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (within 5 μg/m³)')
    plt.title('Training and Test Accuracy Curve')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('./runs/metrics_curves.png')
    plt.show()
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_inputs = torch.tensor(X_test, dtype=torch.float32)
        test_preds = model(test_inputs).squeeze().numpy()
        print(f"Test Predictions: {test_preds}, Actual: {y_test}")
    
    # Save model and scaler
    torch.save(model.state_dict(), './model.pth')
    joblib.dump(scaler, './scaler.pkl')
    
    return model, scaler, train_losses, test_losses, train_accuracies, test_accuracies

# Main execution
if __name__ == "__main__":
    # Define file paths
    air_quality_path = './dataset/air_quality_guting_combined_daily_average.csv'
    rainfall_path = './dataset/taipei_rainfall_combined.csv'
    weather_path = './dataset/taipei_weather_combined.csv'
    
    # Train the model
    model, scaler, train_losses, test_losses, train_accuracies, test_accuracies = train_model(
        air_quality_path, rainfall_path, weather_path
    )