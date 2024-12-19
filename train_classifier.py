import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Load the data
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Extract data and labels
data_list = data_dict['data']
labels = np.asarray(data_dict['labels'])

# Ensure consistent feature vector length
X = np.array([
    sample[:84] if len(sample) > 84 else 
    sample + [0.0] * (84 - len(sample)) 
    for sample in data_list
])
y = labels

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Create preprocessing pipeline with SVM
pipeline = make_pipeline(
    StandardScaler(),
    SVC(kernel='rbf', class_weight='balanced', probability=True)
)

# Fit the model
pipeline.fit(X_train, y_train)

# Predictions and evaluation
y_pred = pipeline.predict(X_test)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix Visualization
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix of Sign Language Classification')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

# Save the model
os.makedirs('./models', exist_ok=True)
with open('./models/sign_language_model.pkl', 'wb') as f:
    pickle.dump({
        'model': pipeline,
        'labels': list(set(y))
    }, f)

print("Model training complete. Model saved to ./models/sign_language_model.pkl")