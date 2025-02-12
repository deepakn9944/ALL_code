#bayes
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy.stats import entropy
import matplotlib.pyplot as plt

# Function to initialize a Naive Bayes model
def initialize_model(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

# Function to create batches from data
def create_batches(x_train, y_train, batch_size):
    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]
        yield x_batch, y_batch

# Function to monitor performance
def monitor_performance(predictions, true_labels):
    if len(predictions) == 0 or len(true_labels) == 0:
        return 0, 0, 0, 0
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='macro')
    precision = precision_score(true_labels, predictions, average='macro')
    recall = recall_score(true_labels, predictions, average='macro')
    return accuracy, f1, precision, recall

# Function to calculate entropy-based disagreement
def calculate_entropy_disagreement(batch_predictions):
    num_samples = batch_predictions.shape[1]
    disagreement_scores = np.zeros(num_samples)

    for i in range(num_samples):
        class_counts = np.bincount(batch_predictions[:, i], minlength=2)  # Assuming binary classification
        disagreement_scores[i] = entropy(class_counts, base=2)  # Base 2 for bits

    return disagreement_scores

# Data preparation
# Assuming `X_initial_scaled`, `y_initial`, `X_main_scaled`, and `y_main` are preloaded data
initial_train_size = int(0.3 * len(X_initial_scaled))
random_indices = np.random.choice(X_initial_scaled.shape[0], initial_train_size, replace=False)

X_initial_train = X_initial_scaled[random_indices]
y_initial_train = y_initial.iloc[random_indices]

X_train, X_test, y_train, y_test = train_test_split(X_main_scaled, y_main, test_size=0.2, random_state=42)

# Train the oracle model
oracle_model = GaussianNB()
oracle_model.fit(X_initial_train, y_initial_train)
oracle_model.fit(X_train, y_train)

print(f"Total samples used for Initial training: {len(X_initial_train)}")
print(f"Total samples used for training: {len(X_train)}")
print(f"Total samples used for testing: {len(X_test)}")

# Define feature subsets for committee models
feature_subsets = [
    list(range(10)),  # First 15 features (indices)
    list(range(10, 20)),  # Next 15 features (indices)
    list(range(20, 30)),
    list(range(30, X_initial_scaled.shape[1]))  # Remaining features (indices)
]

# Initialize models for the committee
models = [initialize_model(X_initial_train[:, subset], y_initial_train) for subset in feature_subsets]

# Function to predict and monitor performance
def predict(X_test, y_test, models):
    batch_predictions = []
    for model, subset in zip(models, feature_subsets):
        X_test_subset = X_test[:, subset]
        batch_predictions.append(model.predict(X_test_subset))
    batch_predictions = np.array(batch_predictions)
    accuracies = [accuracy_score(y_test, pred) for pred in batch_predictions]
    weights = accuracies / np.sum(accuracies)  # Normalize weights
    final_predictions = np.zeros_like(batch_predictions[0], dtype=float)
    for j in range(batch_predictions.shape[0]):
        final_predictions += weights[j] * (batch_predictions[j] == 1)
    final_predictions = np.round(final_predictions).astype(int)
    accuracy, f1, precision, recall = monitor_performance(final_predictions, y_test)
    return accuracy, f1, precision, recall


accuracy, f1, precision, recall = predict(X_test, y_test, models)
print(f"Accuracy: {accuracy*100:.2f}%, F1: {f1:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")
# Iterating through batches for evaluation
num_batches = 10
Total_prediction = 0
batch_size = len(X_train) // num_batches
for i, batch in enumerate(create_batches(X_train, y_train, batch_size)):
    X_batch, y_batch = batch
    if i == 10 or len(X_batch) == 0:
        continue

    # Make predictions with each model using the appropriate features
    batch_predictions = []
    for model, subset in zip(models, feature_subsets):
        batch_subset = X_batch[:, subset]
        batch_predictions.append(model.predict(batch_subset))
    batch_predictions = np.array(batch_predictions)
    accuracy, f1, precision, recall = predict(X_batch, y_batch, models)
    print(f"Initial Test => Accuracy: {accuracy*100:.2f}%, F1: {f1:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")
    Total_prediction +=accuracy

    # Find samples with high and low entropy-based disagreement
    disagreement_scores = calculate_entropy_disagreement(batch_predictions)
    num_samples_to_select = max(1, int(0.05 * len(disagreement_scores)))  # Top 5% high and low

    num_samples_to_select_low = max(1, int(0.05 * len(disagreement_scores)))  # Top 5% high and low
    # Top 10% high entropy
    top_high_entropy_indices = np.argsort(disagreement_scores)[-num_samples_to_select:]
    top_high_entropy_samples = X_batch[top_high_entropy_indices]
    top_high_entropy_labels = oracle_model.predict(top_high_entropy_samples)

    # Top 10% low entropy
    top_low_entropy_indices = np.argsort(disagreement_scores)[:num_samples_to_select_low]
    top_low_entropy_samples = X_batch[top_low_entropy_indices]
    top_low_entropy_labels = oracle_model.predict(top_low_entropy_samples)

    # Combine high and low entropy samples
    selected_samples = np.vstack([top_high_entropy_samples, top_low_entropy_samples])
    selected_labels = np.concatenate([top_high_entropy_labels, top_low_entropy_labels])

    # Update training data and retrain models
    X_initial_train = selected_samples
    y_initial_train = selected_labels

    models = [initialize_model(X_initial_train[:, subset], y_initial_train) for subset in feature_subsets]
    accuracy, f1, precision, recall = predict(X_test, y_test, models)
    print(f"Final Test => Accuracy: {accuracy*100:.2f}%, F1: {f1:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")

accuracy, f1, precision, recall = predict(X_test, y_test, models)
print(f"Final Test => Accuracy: {accuracy*100:.2f}%, F1: {f1:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")
