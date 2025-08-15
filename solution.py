import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import re

# --- Step 1: Load the datasets ---
print("Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# --- Step 2: Preprocess the Text Data ---
print("Cleaning text data...")
def clean_text(text):
    text = str(text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra spaces
    text = ' '.join(text.split())
    return text

train_df['Text'] = train_df['Text'].apply(clean_text)
test_df['Text'] = test_df['Text'].apply(clean_text)

# --- Step 3: Split Data for Validation and Evaluate the Model ---
# Splitting the training data to test our model locally
X = train_df['Text']
y = train_df['Subject']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the pipeline
model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('classifier', MultinomialNB())
])

# Train the model on the training split
print("Training model on a subset of data for validation...")
model_pipeline.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = model_pipeline.predict(X_val)

# Calculate and print the F1-score as per competition criteria
f1_macro_score = f1_score(y_val, y_pred_val, average='macro')
final_score = 100 * f1_macro_score
print(f"Validation F1-score (macro): {f1_macro_score:.4f}")
print(f"Final Score (as per competition): {final_score:.2f}")

# --- Step 4: Final Training and Submission ---
# For the final submission, we train on the entire training dataset
print("\nRetraining model on the full training data for final predictions...")
model_pipeline.fit(X, y)

# Make final predictions on the test data
predictions = model_pipeline.predict(test_df['Text'])

# Create the submission DataFrame
submission_df = pd.DataFrame({
    'ID': test_df['ID'],
    'Subject': predictions
})

# Save the submission DataFrame to a CSV file
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' created successfully!")
print("First 5 rows of the generated submission file:")
print(submission_df.head())
