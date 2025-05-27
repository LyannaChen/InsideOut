import pandas as pd
from collections import Counter
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation

# Load the uploaded CSV file
def load_data():
    files = [
        './generated_scripts/llama/generated_scripts.csv',
        './generated_scripts/chatgpt/generated_scripts.csv',
        './generated_scripts/mistral/generated_scripts.csv',
        './generated_scripts/qwen/generated_scripts.csv'
    ]
    df_all = [pd.read_csv(file) for file in files]
    return pd.concat(df_all, ignore_index=True)

# Prepare texts grouped by culture
def preprocess(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    return text

# log-odds ratio analysis for lexical saliency
def compute_log_odds_ratio(culture_texts, other_texts, alpha=0.01):
    """
    Compute log-odds ratio with an informative Dirichlet prior for a target culture
    against all other cultures combined.

    Args:
        culture_texts (List[List[str]]): Tokenized documents for the target culture
        other_texts (List[List[str]]): Tokenized documents for all other cultures
        alpha (float): Smoothing parameter (Dirichlet prior)

    Returns:
        List[Tuple[str, float, float]]: Each entry contains (word, log-odds, z-score),
                                        sorted by descending z-score
    """
    culture_counts = Counter([w for doc in culture_texts for w in doc])
    other_counts = Counter([w for doc in other_texts for w in doc])
    vocab = set(culture_counts.keys()).union(other_counts.keys())

    results = []
    for word in vocab:
        a = culture_counts[word] + alpha
        b = other_counts[word] + alpha
        log_odds = np.log(a / b)
        variance = 1 / (a + alpha) + 1 / (b + alpha)
        z_score = log_odds / np.sqrt(variance)
        results.append((word, log_odds, z_score))

    # Sort by z-score descending to get most salient words
    sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
    return sorted_results[:20]  # Return top 20

# Topic modeling using sklearn
def run_sklearn_lda(culture_texts_dict, num_topics=1, num_words=5, min_docs=2):
    topic_results = {}
    
    for culture, docs in culture_texts_dict.items():
        if len(docs) < min_docs:
            topic_results[culture] = None
            continue

        # Vectorize
        vectorizer = CountVectorizer(stop_words='english')
        X = vectorizer.fit_transform(docs)

        # Run LDA
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda.fit(X)

        # Get top words for the topic
        words = vectorizer.get_feature_names_out()
        topic = lda.components_[0]
        top_words = [words[i] for i in topic.argsort()[:-num_words-1:-1]]
        topic_results[culture] = ", ".join(top_words)

    return topic_results


def main():
    stop_words = set(ENGLISH_STOP_WORDS)
    df_all = load_data()

    # Initialize dictionary to store preprocessed texts by country
    data = {}

    # Group by country and preprocess the 'generated' text
    for country, group in df_all.groupby('country'):
        texts = group['generated'].dropna().astype(str).tolist()
        processed = [
            re.sub(r"[^a-zA-Z0-9\s]", "", doc.lower()).split() for doc in texts
        ]
        processed = [[word for word in doc if word not in stop_words] for doc in processed]
        data[country] = processed

    # Compute log-odds ratio
    log_odds_results = {}

    for culture in data:
        other_corpus = [doc for c, docs in data.items() if c != culture for doc in docs]
        log_odds_results[culture] = compute_log_odds_ratio(data[culture], other_corpus)

    # Preprocess full texts for LDA
    df_all['cleaned'] = df_all['generated'].astype(str).apply(preprocess)
    culture_texts_dict = df_all.groupby('country')['cleaned'].apply(list).to_dict()

    topics = run_sklearn_lda(culture_texts_dict)

    # Display results
    print("\n=== Log-Odds Ratio Analysis ===")
    for culture in log_odds_results.keys():
        print(culture, ': ', ', '.join([x[0] for x in log_odds_results[culture]]))

    print("\n=== Topic Modeling Results ===")
    for culture, topic in topics.items():
        print(f"{culture}: {topic}")

if __name__ == "__main__":
    main()