import csv
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
# Read the CSV file
with open('cnbc_headlines.csv', 'r') as f:
    reader = csv.DictReader(f)
    data = [(row['Headlines'], row['Time'], row['Description']) for row in reader]
# Remove stopwords
stop_words = set(stopwords.words('english'))
# Stem the words
ps = PorterStemmer()
preprocessed_data = []
for headline, time, description in data:
    words = word_tokenize(description)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    stemmed_words = [ps.stem(word) for word in filtered_words]
    preprocessed_text = ' '.join(stemmed_words)
    preprocessed_data.append((preprocessed_text, headline))

# Convert the text data into numerical features
X = [text for text, label in preprocessed_data]
y = [label for text, label in preprocessed_data]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a machine learning model
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Evaluate the performance of the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)