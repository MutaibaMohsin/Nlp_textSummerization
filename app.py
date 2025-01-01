import streamlit as st
import pandas as pd
from summarizer import summarize_text
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np
from nltk.tokenize import word_tokenize
import seaborn as sns
import matplotlib.pyplot as plt

# Title
st.title("Text Summarization Tool with Evaluation")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file with 'Description' and 'Ground Truth' columns", type=["csv"])

if uploaded_file:
    # Load the dataset
    data = pd.read_csv(uploaded_file)

    if "Description" not in data.columns or "Ground Truth" not in data.columns:
        st.error("The uploaded file must contain 'Description' and 'Ground Truth' columns.")
    else:
        # Display dataset preview
        st.subheader("Dataset Preview")
        st.write(data.head())

        # Summarization
        st.subheader("Summarized Descriptions")
        num_sentences = st.slider("Select the number of sentences in the summary", 1, 5, 2)

        summaries = []
        for description in data["Description"]:
            summary = summarize_text(description, num_sentences=num_sentences)
            summaries.append(summary)

        data["Summary"] = summaries
        st.write(data[["Description", "Ground Truth", "Summary"]])

        # Download summarized data
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df_to_csv(data)
        st.download_button(
            label="Download Summarized Data",
            data=csv,
            file_name="summarized_descriptions.csv",
            mime="text/csv"
        )

        # Evaluation Section
        st.subheader("Evaluation Metrics")

        # Tokenize summaries and ground truth
        generated_tokens = [set(word_tokenize(summary.lower())) for summary in data["Summary"]]
        ground_truth_tokens = [set(word_tokenize(gt.lower())) for gt in data["Ground Truth"]]

        # Calculate precision, recall, F1-score
        def calculate_metrics(predictions, references):
            precision_list = []
            recall_list = []
            f1_list = []
            for pred, ref in zip(predictions, references):
                tp = len(pred & ref)
                fp = len(pred - ref)
                fn = len(ref - pred)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)
            return np.mean(precision_list), np.mean(recall_list), np.mean(f1_list)

        precision, recall, f1 = calculate_metrics(generated_tokens, ground_truth_tokens)
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"F1-Score: {f1:.2f}")

        # Additional analysis (true positives, false positives, false negatives)
        tp = sum(len(pred & ref) for pred, ref in zip(generated_tokens, ground_truth_tokens))
        fp = sum(len(pred - ref) for pred, ref in zip(generated_tokens, ground_truth_tokens))
        fn = sum(len(ref - pred) for pred, ref in zip(generated_tokens, ground_truth_tokens))
        st.write(f"True Positives (tokens in both): {tp}")
        st.write(f"False Positives (extra tokens): {fp}")
        st.write(f"False Negatives (missing tokens): {fn}")

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        all_tokens = list(set.union(*generated_tokens, *ground_truth_tokens))
        y_true = [int(token in ref) for ref in ground_truth_tokens for token in all_tokens]
        y_pred = [int(token in pred) for pred in generated_tokens for token in all_tokens]

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["False", "True"], yticklabels=["False", "True"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
