import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from rich import print
from rich.console import Console
from rich.table import Table

console = Console()

# Step 1: Load Standard Dataset
def load_standard_dataset(file_path):
    """
    Load the standard dataset from an Excel file.
    Columns: 'CAN_ID' and 'Frequency'.
    """
    try:
        standard_data = pd.read_excel(file_path)
        console.log(f"Columns in the dataset: [cyan]{list(standard_data.columns)}[/cyan]")

        # Standardize column names
        standard_data.columns = standard_data.columns.str.strip().str.upper()

        # Check required columns
        required_columns = ['CAN ID', 'FREQUENCY (HZ)']
        for col in required_columns:
            if col not in standard_data.columns:
                raise ValueError(f"Expected column '{col}' not found in dataset.")

        # Rename columns for consistency
        standard_data = standard_data.rename(columns={'CAN ID': 'CAN_ID', 'FREQUENCY (HZ)': 'Frequency'})
        console.log(f"Standard dataset loaded from [bold cyan]{file_path}[/bold cyan].")
        return standard_data[['CAN_ID', 'Frequency']]
    except Exception as e:
        console.log(f"[bold red]Error loading dataset: {e}[/bold red]")
        raise

# Step 2: Generate Synthetic CAN Data
def generate_can_data(num_samples=10000):
    """
    Generate synthetic CAN data with CAN_ID and Frequency.
    """
    np.random.seed(42)
    can_ids = np.random.choice(['0x101', '0x102', '0x103', '0x104', '0x105'], size=num_samples)
    frequencies = np.random.randint(1, 100, size=num_samples)  # Random frequencies between 1 and 100
    data = pd.DataFrame({'CAN_ID': can_ids, 'Frequency': frequencies})
    console.log(f"Synthetic CAN data with {num_samples} samples generated.")
    return data

# Step 3: Compare and Use Dataset
def use_appropriate_dataset(generated_data, standard_data):
    """
    Compare the generated dataset with the standard dataset size.
    Use the standard dataset if the generated dataset size exceeds 4x the standard dataset size.
    """
    if len(generated_data) > 4 * len(standard_data):
        console.log(f"[bold yellow]Generated dataset size exceeds 4 times the standard dataset size.[/bold yellow]")
        console.log("[bold green]Using the standard dataset instead.[/bold green]")
        return standard_data
    console.log("[bold green]Using the generated dataset.[/bold green]")
    return generated_data

# Step 4: Preprocess CAN Data
def preprocess_can_data(data):
    """
    Preprocess CAN data by converting CAN IDs to integers and scaling frequencies.
    """
    data['CAN_ID'] = data['CAN_ID'].apply(lambda x: int(x, 16))  # Convert hex CAN_IDs to integers
    scaler = MinMaxScaler()
    data['Frequency'] = scaler.fit_transform(data[['Frequency']])
    can_ids = data['CAN_ID'].values
    frequencies = data['Frequency'].values
    return can_ids, frequencies, scaler

# Step 5: Build the Deep Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Step 6: Train the Autoencoder
def train_autoencoder(autoencoder, frequencies, epochs=20):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    frequencies_tensor = torch.tensor(frequencies, dtype=torch.float32).unsqueeze(1)

    for epoch in range(epochs):
        autoencoder.train()
        optimizer.zero_grad()
        outputs = autoencoder(frequencies_tensor)
        loss = criterion(outputs, frequencies_tensor)
        loss.backward()
        optimizer.step()
        console.log(f"[green]Epoch {epoch+1}/{epochs}[/green], Loss: {loss.item():.4f}")

# Step 7: Detect Anomalies
def detect_anomalies(autoencoder, frequencies, threshold):
    """
    Detect anomalies in the test data.
    Calculates reconstruction error for each test sample and flags anomalies.
    """
    autoencoder.eval()
    frequencies_tensor = torch.tensor(frequencies, dtype=torch.float32).unsqueeze(1)

    # Perform forward pass to reconstruct
    with torch.no_grad():
        reconstructed = autoencoder(frequencies_tensor).numpy()

    # Calculate reconstruction error for each sample
    reconstruction_error = (frequencies - reconstructed.squeeze()) ** 2  # Element-wise squared error

    # Identify anomalies
    anomalies = reconstruction_error > threshold
    return reconstruction_error, anomalies

# Visualization Functions
def visualize_reconstruction_error(reconstruction_error, threshold):
    """
    Visualize the distribution of reconstruction errors.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(reconstruction_error, bins=50, alpha=0.7, label='Reconstruction Error')
    plt.axvline(x=threshold, color='red', linestyle='--', label='Threshold')
    plt.title("Reconstruction Error Distribution")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_anomaly_detection(frequencies, anomalies):
    """
    Visualize anomalies in the dataset.
    """
    plt.figure(figsize=(10, 6))
    normal_data = np.where(~anomalies)
    anomaly_data = np.where(anomalies)
    plt.scatter(normal_data[0], frequencies[normal_data], c='blue', label='Normal', alpha=0.7)
    plt.scatter(anomaly_data[0], frequencies[anomaly_data], c='red', label='Anomaly', alpha=0.7)
    plt.title("Anomaly Detection Results")
    plt.xlabel("Sample Index")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()

def evaluate_performance(ground_truth, predicted_labels):
    """
    Evaluate model performance with precision, recall, and F1-score.
    Display results in a styled table.
    """
    precision = precision_score(ground_truth, predicted_labels)
    recall = recall_score(ground_truth, predicted_labels)
    f1 = f1_score(ground_truth, predicted_labels)

    table = Table(title="Performance Metrics")
    table.add_column("Metric", justify="left", style="cyan")
    table.add_column("Value", justify="right", style="magenta")
    table.add_row("Precision", f"{precision:.2f}")
    table.add_row("Recall", f"{recall:.2f}")
    table.add_row("F1-Score", f"{f1:.2f}")

    console.print(table)

# Main Pipeline
if __name__ == "__main__":
    console.rule("[bold green]Step 1: Loading Standard Dataset[/bold green]")
    standard_dataset_file = "random_can_data.xlsx"
    standard_data = load_standard_dataset(standard_dataset_file)

    console.rule("[bold green]Step 2: Generating Synthetic CAN Data[/bold green]")
    generated_data = generate_can_data(10000)

    console.rule("[bold green]Step 3: Comparing Datasets[/bold green]")
    final_data = use_appropriate_dataset(generated_data, standard_data)

    console.rule("[bold green]Step 4: Preprocessing Data[/bold green]")
    can_ids, frequencies, scaler = preprocess_can_data(final_data)
    console.log("Data preprocessing completed.")

    console.rule("[bold green]Step 5: Splitting Data[/bold green]")
    can_ids_train, can_ids_test, frequencies_train, frequencies_test = train_test_split(
        can_ids, frequencies, test_size=0.2, random_state=42
    )
    console.log("Data split into training and testing sets.")

    console.rule("[bold green]Step 6: Training Autoencoder[/bold green]")
    autoencoder = Autoencoder(input_dim=1)
    train_autoencoder(autoencoder, frequencies_train, epochs=20)

    console.rule("[bold green]Step 7: Detecting Anomalies[/bold green]")
    threshold = 0.01
    reconstruction_error, anomalies = detect_anomalies(autoencoder, frequencies_test, threshold)

    console.rule("[bold green]Step 8: Visualizing Results[/bold green]")
    visualize_reconstruction_error(reconstruction_error, threshold)
    visualize_anomaly_detection(frequencies_test, anomalies)

    console.rule("[bold green]Step 9: Confusion Matrix and Performance Metrics[/bold green]")
    # Simulated ground truth (replace with actual labels if available)
    ground_truth = [0] * int(0.9 * len(anomalies)) + [1] * int(0.1 * len(anomalies))

    if len(ground_truth) != len(anomalies):
        raise ValueError("Ground truth labels and predicted anomalies must have the same length!")

    predicted_labels = [1 if anomaly else 0 for anomaly in anomalies]
    cm = confusion_matrix(ground_truth, predicted_labels, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomaly"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    evaluate_performance(ground_truth, predicted_labels)
