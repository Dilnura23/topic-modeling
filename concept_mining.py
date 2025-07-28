
import pandas as pd
import numpy as np
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
import nltk
import spacy
from nltk.stem import WordNetLemmatizer

# Download necessary resources
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords

# Load spaCy for noun phrase extraction and Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()

# Preprocessing: Improved version with lemmatization and removal of non-alphabetic words
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    stop_words = set(stopwords.words('english'))
    words = [lemmatizer.lemmatize(word) for word in text.split() if len(word) > 2 and word.isalpha() and word not in stop_words]
    return " ".join(words)

# Extract noun phrases using spaCy (more semantic meaning)
def extract_phrases(text):
    doc = nlp(text)
    return ['_'.join(token.text.lower() for token in chunk if token.text not in string.punctuation) 
            for chunk in doc.noun_chunks]

# Truncate long documents to reduce memory usage and focus on the most relevant terms
def truncate_tokens(tokens, max_len=100):
    return tokens[:max_len]

# Find the best number of topics using coherence score
def get_best_lda_model(tokenized_docs, dictionary, corpus, min_topics=4, max_topics=12):
    best_model = None
    best_score = -1
    for n_topics in range(min_topics, max_topics + 1):
        model = models.LdaModel(corpus, num_topics=n_topics, id2word=dictionary, passes=25, iterations=500, random_state=42)
        coherence = CoherenceModel(model=model, texts=tokenized_docs, dictionary=dictionary, coherence='c_v').get_coherence()
        print(f"Topics: {n_topics}, Coherence Score: {coherence:.4f}")
        if coherence > best_score:
            best_model, best_score = model, coherence
    return best_model

# Hybrid concept extraction with more attention to top words in topics
def get_document_concepts(lda_model, corpus, tfidf_matrix, tfidf_vectorizer, top_n=5):
    feature_names = tfidf_vectorizer.get_feature_names_out()
    concepts_per_doc = []

    for doc_idx, bow in enumerate(corpus):
        topic_dist = lda_model.get_document_topics(bow)
        top_topic = sorted(topic_dist, key=lambda x: x[1], reverse=True)[0][0]
        topic_words = lda_model.get_topic_terms(topicid=top_topic, topn=20)

        topic_keywords = [(lda_model.id2word[word_id], prob) for word_id, prob in topic_words]

        tfidf_row = tfidf_matrix.getrow(doc_idx)
        tfidf_dict = {feature_names[i]: tfidf_row[0, i] for i in tfidf_row.indices}

        hybrid_scores = {}
        for word, lda_prob in topic_keywords:
            tfidf_score = tfidf_dict.get(word, 1e-6)  # smoothing for unseen words
            hybrid_scores[word] = lda_prob * tfidf_score

        # Sort and select top concepts based on the hybrid score
        top_concepts = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        concepts_per_doc.append({w: round(score, 4) for w, score in top_concepts})

    return concepts_per_doc

def main():
    try:
        print("📥 Reading Excel file...")
        df = pd.read_excel('Copy of _Streamlined Dataset_pF.xlsx')

        # Only work with the first 5 rows -- CHANGEd because it takes a lot of time to process 600 articles
        df = df.head(5)

        print("🧹 Preprocessing and phrase extraction...")
        documents = []
        for _, row in df.iterrows():
            title = preprocess_text(row['Title'] if pd.notna(row['Title']) else '')
            abstract = preprocess_text(row['Abstract Note'] if pd.notna(row['Abstract Note']) else '')
            full_text = f"{title} {abstract}"
            phrases = extract_phrases(full_text)
            phrases = truncate_tokens(phrases)
            documents.append(" ".join(phrases))

        print("🧠 TF-IDF vectorization...")
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
        tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

        print("📊 Tokenizing for LDA...")
        tokenized_docs = [doc.split() for doc in documents]
        dictionary = corpora.Dictionary(tokenized_docs)
        corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

        print("🔍 Finding best LDA model...")
        lda_model = get_best_lda_model(tokenized_docs, dictionary, corpus)

        print("📌 Extracting hybrid concepts...")
        hybrid_concepts = get_document_concepts(lda_model, corpus, tfidf_matrix, tfidf_vectorizer, top_n=5)
        top_topics = [sorted(lda_model.get_document_topics(bow), key=lambda x: -x[1])[0][0] for bow in corpus]

        print("💾 Saving results...")
        df['Top Topic'] = top_topics
        df['Hybrid Key Concepts'] = [list(concepts.keys()) for concepts in hybrid_concepts]
        df['Key Concepts (Weighted)'] = [concepts for concepts in hybrid_concepts]
        df.to_excel('hybrid_concepts_output_limited.xlsx', index=False)
        print("✅ Done: Saved to 'hybrid_concepts_output_limited.xlsx'.")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
