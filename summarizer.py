import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import string


# Download required NLTK data files
nltk.download('punkt')
nltk.download('stopwords')


# Summarizer function
def summarize_text(text, num_sentences=1):
    if not text or not isinstance(text, str):  # Handle invalid input
        return "No valid description provided."

    # Preprocessing
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]

    if not words:  # Handle edge cases with no valid words
        return "Description contains no meaningful content."

    # Calculate word frequencies
    word_frequencies = Counter(words)
    max_freq = max(word_frequencies.values())
    for word in word_frequencies:
        word_frequencies[word] /= max_freq

    print("Word Frequencies:", word_frequencies)  # Debugging

    # Score sentences
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return " ".join(sentences)  # Return original text if it's short

    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_frequencies:
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = word_frequencies[word]
                else:
                    sentence_scores[sentence] += word_frequencies[word]

    print("Sentence Scores:", sentence_scores)  # Debugging

    # Select top sentences
    sorted_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    print("Sorted Sentences:", sorted_sentences)  # Debugging

    top_sentences = sorted(sorted_sentences[:num_sentences], key=sentences.index)
    print("Top Sentences:", top_sentences)  # Debugging

    return " ".join(top_sentences)
