import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle
import ast  # To parse embedding strings from CSV

# Load embeddings from CSV
def load_embeddings(csv_path):
    df = pd.read_csv(csv_path)
    # Convert string embeddings to numpy arrays
    embeddings = np.array([np.array(ast.literal_eval(embed)) for embed in df['embedding']])
    labels = df['person_name'].values
    return embeddings, labels

# Load train and validation data
train_embeddings, train_labels = load_embeddings('embeddings_train.csv')
val_embeddings, val_labels = load_embeddings('embeddings_val.csv')

print(f"Training samples: {len(train_embeddings)}")
print(f"Validation samples: {len(val_embeddings)}")
print(f"Unique persons: {len(np.unique(train_labels))}")

# Train SVM
svm = SVC(kernel='linear', probability=True, random_state=42)
svm.fit(train_embeddings, train_labels)

# Predict on training set (for overfitting check)
train_preds = svm.predict(train_embeddings)
train_accuracy = accuracy_score(train_labels, train_preds)
print(f"Training accuracy: {train_accuracy:.4f}")

# Predict on validation set
val_preds = svm.predict(val_embeddings)
val_accuracy = accuracy_score(val_labels, val_preds)
print(f"Validation accuracy: {val_accuracy:.4f}")
print("\nClassification Report (Validation):")
print(classification_report(val_labels, val_preds))

# Save the trained model
with open('face_recognizer.pkl', 'wb') as f:
    pickle.dump(svm, f)
print("✅ SVM model saved to face_recognizer.pkl")