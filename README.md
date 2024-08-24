# Fake News Detection Using Machine Learning

This project is a simple implementation of a fake news detection model using Python, scikit-learn, and the PassiveAggressiveClassifier. The model can classify news articles as either **REAL** or **FAKE** based on their content.

## Dataset

The project uses two datasets:
- **true.csv**: Contains true news articles.
- **fake.csv**: Contains fake news articles.

## Project Structure

- `true.csv`: CSV file containing true news articles.
- `fake.csv`: CSV file containing fake news articles.
- `fake_news_detection.py`: The main Python script for training and testing the model.
- `model.pkl`: The saved machine learning model (optional, created after running the script).
- `vectorizer.pkl`: The saved TF-IDF vectorizer (optional, created after running the script).
- `README.md`: Project documentation.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/fake-news-detection.git
    cd fake-news-detection
    ```

2. Install the required Python packages:
    ```bash
    pip install pandas numpy scikit-learn
    ```

3. Add your datasets (`true.csv` and `fake.csv`) to the project directory.

## Usage

1. Run the `index.py` script to train the model:
    ```bash
    python fake_news_detection.py
    ```

2. The script will:
   - Combine the `true.csv` and `fake.csv` datasets.
   - Train a `PassiveAggressiveClassifier` to detect fake news.
   - Display the model's accuracy and confusion matrix.
   - Save the trained model and vectorizer to `model.pkl` and `vectorizer.pkl`.

3. To predict whether a new piece of text is real or fake, use the `predict_fake_news` function in the script:
    ```python
    example_text = "Your article text here"
    prediction = predict_fake_news(example_text)
    print(f"Prediction: {prediction}")
    ```

## Example Output

After running the script, you should see output similar to:

```plaintext
Datasets loaded successfully.
Datasets combined.
Dataset shuffled.
(44898, 5)
                 title  ... label
0  You Can Smell It: Donald Trump Jr....  ...  REAL
1  Is It An NFL Sunday If Odell Beckham...  ...  FAKE

Accuracy: 92.74%
Confusion Matrix:
[[5897  303]
 [ 334 5829]]
Model and vectorizer saved.
