# Vehicle Anti-Theft: Text Classification Solution

This project provides a complete machine learning solution for a text classification problem. The goal is to classify vehicle-related text into different categories or "subjects."

## 1. Problem Statement

The challenge is a multi-class text classification task where the objective is to predict the correct `Subject` for a given text snippet. The solution is evaluated using the **f1_score**, with a specific emphasis on performance across all classes, as indicated by the **macro-average** calculation. The final score is a percentage: `100 * f1_score(actual, predicted, average='macro')`.

## 2. Methodology

The solution follows a standard machine learning workflow, focusing on effective data preprocessing and a robust classification model.

* **Data Preprocessing**: Raw text data was prepared for the model by:
    * **Cleaning**: Removing special characters, numbers, and converting all text to lowercase to ensure consistency.
    * **Vectorization**: Using **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert the cleaned text into a numerical format. TF-IDF highlights words that are important to a specific document but not common across the entire dataset.

* **Model Training**: A **Multinomial Naive Bayes** classifier was chosen due to its efficiency and effectiveness on text data. The entire process—from vectorization to classification—was streamlined using a **scikit-learn Pipeline**.

* **Evaluation**: The model's performance was evaluated on a held-out **validation set** using the specified **macro F1-score**. The model was then retrained on the entire dataset for final predictions.

## 3. How to Run the Code

1.  **Clone the repository:**
    ```
    git clone [https://github.com/your_username/your_project_name.git](https://github.com/your_username/your_project_name.git)
    cd your_project_name
    ```

2.  **Place the data files:** Ensure `train.csv` and `test.csv` are in the same directory as the Python script.

3.  **Install dependencies:**
    ```
    pip install pandas scikit-learn
    ```

4.  **Run the script:**
    ```
    python solution.py
    ```
    This will generate the `submission.csv` file.

## 4. Final Submission File

The script will produce a `submission.csv` file containing the predicted subjects for the test data, ready for submission to the competition.
