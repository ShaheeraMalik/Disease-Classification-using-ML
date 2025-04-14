import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Disease Clustering and Evaluation", layout="wide")
st.title("🧬 Disease Clustering and ML Evaluation")

uploaded_features = st.file_uploader("Upload disease_features.csv", type=["csv"])
uploaded_onehot = st.file_uploader("Upload encoded_output2.csv", type=["csv"])

if uploaded_features and uploaded_onehot:
    df = pd.read_csv(uploaded_features)
    one_hot = pd.read_csv(uploaded_onehot)

    cols_to_parse = ['Risk Factors', 'Symptoms', 'Signs', 'Subtypes']
    for col in cols_to_parse:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
        df[col + '_str'] = df[col].apply(lambda x: ' '.join(x))

    tfidf_risk = TfidfVectorizer().fit_transform(df['Risk Factors_str'])
    tfidf_symptoms = TfidfVectorizer().fit_transform(df['Symptoms_str'])
    tfidf_signs = TfidfVectorizer().fit_transform(df['Signs_str'])
    tfidf_combined = hstack([tfidf_risk, tfidf_symptoms, tfidf_signs])

    sparsity_tfidf = 1.0 - (tfidf_combined.count_nonzero() / (tfidf_combined.shape[0] * tfidf_combined.shape[1]))
    sparsity_onehot = 1.0 - (np.count_nonzero(one_hot.values) / one_hot.size)

    st.subheader("📊 Sparsity & Feature Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Sparsity")
        st.metric(label="TF-IDF Sparsity", value=f"{sparsity_tfidf:.4f}")
        st.metric(label="One-Hot Sparsity", value=f"{sparsity_onehot:.4f}")

    with col2:
        st.markdown("#### Feature Count")
        st.metric(label="TF-IDF Features", value=tfidf_combined.shape[1])
        st.metric(label="One-Hot Features", value=one_hot.shape[1])


    # --- Dimensionality Reduction ---
    one_hot_numeric = one_hot.select_dtypes(include=[np.number])
    pca = PCA(n_components=3).fit(one_hot_numeric)
    svd = TruncatedSVD(n_components=3).fit(tfidf_combined)
    pca_result = pca.transform(one_hot_numeric)
    svd_result = svd.transform(tfidf_combined)

    disease_category_map = {
        'Asthma': 'Respiratory', 'COPD': 'Respiratory', 'Pneumonia': 'Respiratory',
        'Pulmonary Embolism': 'Respiratory', 'Stroke': 'Neurological', 'Epilepsy': 'Neurological',
        'Migraine': 'Neurological', 'Multiple Sclerosis': 'Neurological', 'Alzheimer': 'Neurological',
        'Heart Failure': 'Cardiovascular', 'Atrial Fibrillation': 'Cardiovascular',
        'Acute Coronary Syndrome': 'Cardiovascular', 'Hypertension': 'Cardiovascular',
        'Aortic Dissection': 'Cardiovascular', 'Cardiomyopathy': 'Cardiovascular',
        'Diabetes': 'Endocrine', 'Hyperlipidemia': 'Endocrine', 'Thyroid Disease': 'Endocrine',
        'Pituitary Disease': 'Endocrine', 'Adrenal Insufficiency': 'Endocrine',
        'Gastritis': 'Gastrointestinal', 'Peptic Ulcer Disease': 'Gastrointestinal',
        'Gastro-oesophageal Reflux Disease': 'Gastrointestinal',
        'Upper Gastrointestinal Bleeding': 'Gastrointestinal', 'Tuberculosis': 'Infectious'
    }
    df['Disease Category'] = df['Disease'].map(disease_category_map).fillna('Other')

    st.subheader("📈 Dimensionality Reduction Visualization")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(x=svd_result[:, 0], y=svd_result[:, 1], hue=df['Disease Category'], palette='tab10', s=60, ax=ax1)
    ax1.set_title("SVD on TF-IDF")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=df['Disease Category'], palette='tab10', s=60, ax=ax2)
    ax2.set_title("PCA on One-Hot")
    st.pyplot(fig2)

    # --- Model Evaluation ---
    y = df['Disease Category']
    tfidf_normalized = normalize(tfidf_combined)
    one_hot_normalized = normalize(one_hot_numeric)

    def evaluate_knn(X, y, k_values, metrics):
        results = []
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for k in k_values:
            for metric_name, metric in metrics.items():
                model = KNeighborsClassifier(n_neighbors=k, metric=metric)
                scores = cross_validate(model, X, y, cv=cv, scoring={
                    'accuracy': make_scorer(accuracy_score),
                    'precision': make_scorer(precision_score, average='weighted'),
                    'recall': make_scorer(recall_score, average='weighted'),
                    'f1': make_scorer(f1_score, average='weighted')
                })
                results.append({
                    'Model': 'KNN', 'K': k, 'Metric': metric_name,
                    'Accuracy': np.mean(scores['test_accuracy']),
                    'Precision': np.mean(scores['test_precision']),
                    'Recall': np.mean(scores['test_recall']),
                    'F1-Score': np.mean(scores['test_f1'])
                })
        return results

    def evaluate_logistic(X, y):
        model = LogisticRegression(max_iter=1000)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_validate(model, X, y, cv=cv, scoring={
            'accuracy': make_scorer(accuracy_score),
            'f1': make_scorer(f1_score, average='weighted')
        })
        return {'Model': 'Logistic Regression', 'Accuracy': np.mean(scores['test_accuracy']), 'F1-Score': np.mean(scores['test_f1'])}

    metrics = {'euclidean': 'euclidean', 'manhattan': 'manhattan', 'cosine': 'cosine'}
    k_values = [3, 5, 7]
    knn_tfidf = evaluate_knn(tfidf_normalized, y, k_values, metrics)
    knn_onehot = evaluate_knn(one_hot_normalized, y, k_values, metrics)

    logreg_tfidf = evaluate_logistic(tfidf_normalized, y)
    logreg_onehot = evaluate_logistic(one_hot_normalized, y)

    st.subheader("📋 Model Performance Summary")
    knn_df = pd.DataFrame(knn_tfidf + knn_onehot)
    st.dataframe(knn_df)

    logreg_df = pd.DataFrame([logreg_tfidf, logreg_onehot])
    st.dataframe(logreg_df)

    st.subheader("📊 Model F1-Score Visualization")
    fig3, ax3 = plt.subplots()
    sns.barplot(data=knn_df, x='K', y='F1-Score', hue='Metric', ax=ax3)
    ax3.set_title("KNN F1-Score by K and Distance")
    st.pyplot(fig3)

    fig4, ax4 = plt.subplots()
    sns.barplot(data=knn_df, x='Metric', y='F1-Score', hue='Model', ax=ax4)
    ax4.set_title("TF-IDF vs One-Hot (KNN)")
    st.pyplot(fig4)

    combined_df = pd.concat([
        knn_df.groupby('Model').agg({'F1-Score': 'mean'}).reset_index().assign(Encoding='KNN'),
        logreg_df.assign(Model='Logistic Regression')
    ])

    fig5, ax5 = plt.subplots()
    sns.barplot(data=combined_df, x='Model', y='F1-Score', ax=ax5)
    ax5.set_title("KNN vs Logistic Regression: F1 Comparison")
    st.pyplot(fig5)

    # --- Report Section ---
    st.subheader("📝 Analytical Report")
    if st.button("Generate Report"):
        st.markdown("""
            # Task 4: Critical Analysis

            ### 1. Why TF-IDF might outperform One-Hot Encoding (or vice versa) for this dataset:

            - **TF-IDF vs One-Hot Encoding:**
            - **One-Hot Encoding** represents categorical variables as binary vectors where each feature corresponds to a specific word or term. For example, in the case of diseases and their associated risk factors, symptoms, or signs, each unique term gets its own feature. If a term is present, it is represented as `1`, otherwise `0`.
            - **TF-IDF (Term Frequency-Inverse Document Frequency)**, on the other hand, adjusts term importance by considering both the term's frequency in the document and its rarity across all documents. Common terms across multiple categories (like "cough" in respiratory diseases) are down-weighted, while terms specific to particular diseases (like "stroke" in neurological diseases) are emphasized.

            - **Why TF-IDF might outperform One-Hot Encoding:**
            1. **Feature Weighting:**
                - TF-IDF places a higher weight on terms that are unique to certain diseases (e.g., "asthma" for respiratory diseases), allowing it to capture more informative and discriminative features. One-hot encoding does not account for term frequency or rarity across categories.
            2. **Dimensionality and Sparsity:**
                - One-Hot Encoding can lead to extremely high-dimensional sparse matrices, especially when dealing with large vocabularies of disease-related terms. The result is often a sparse matrix, where most entries are zeros. TF-IDF, though still sparse, offers more meaningful representations by prioritizing important features and reducing the impact of overly common terms.
            3. **Better Representation of Context:**
                - TF-IDF captures context better than one-hot encoding, which only reflects the presence or absence of a term. It enables a more nuanced understanding of the relationship between diseases and their associated features.

            - **Why One-Hot Encoding might outperform TF-IDF:**
            1. **Simple Structure:**
                - One-hot encoding is easy to interpret and works well when the features have clear-cut boundaries. In certain cases, especially when disease categories are well-defined and terms do not overlap much across categories, one-hot encoding might capture sufficient information.
            2. **Low Variance in Features:**
                - If the terms involved are consistently used across the dataset, one-hot encoding may perform similarly to TF-IDF without the complexity of calculating term frequency-inverse document frequency. For example, diseases that have a very consistent pattern of associated symptoms may benefit from one-hot encoding's simplicity.
            
            ### 2. Clinical Relevance of the Results:

            - **Do TF-IDF clusters align with real-world disease categories?**
            - The results from TF-IDF feature extraction can often be more aligned with clinical realities, especially when using domain-specific text data. For instance, terms like "pulmonary embolism" or "stroke" are highly distinctive and should map well to their respective disease categories (e.g., Cardiovascular or Neurological). By giving higher weight to such specific terms, TF-IDF allows the model to focus on the unique characteristics of diseases.
            - The clustering of diseases based on TF-IDF features might reflect how diseases are categorized in medical practice (e.g., **Respiratory**, **Neurological**, **Cardiovascular**). Diseases within these categories share common risk factors and symptoms, which should be captured effectively by TF-IDF.
            
            - **Interpretation of Disease Clusters:**
            - **Respiratory Diseases** like asthma, pneumonia, and COPD, which share symptoms such as shortness of breath and cough, should cluster together. TF-IDF should be able to distinguish these terms and give greater importance to unique disease-specific symptoms.
            - **Neurological Diseases** like epilepsy, stroke, and Alzheimer's should also form distinct clusters with their unique symptoms and risk factors (e.g., neurological impairment, headaches). The model’s performance with TF-IDF might reflect the hierarchical nature of disease categorization.
            - By contrast, **One-Hot Encoding** could struggle to capture this nuanced clustering as it does not differentiate between frequent and rare terms.

            ### 3. Limitations of Both Encoding Methods:

            - **Limitations of One-Hot Encoding:**
            1. **High Dimensionality:**
                - One-hot encoding can lead to very high-dimensional datasets when dealing with large vocabularies, leading to sparse matrices with many zero values. This increases computational costs and can hinder the performance of models due to the "curse of dimensionality."
            2. **Lack of Feature Weighting:**
                - One-hot encoding does not consider term importance or frequency, which means that all terms are treated equally, even if some terms are more relevant or discriminative for classification tasks. It does not capture any relationships or context between features.
            3. **Overfitting Risk:**
                - Due to the high dimensionality, the model may overfit, especially if the dataset is small. With too many features and not enough samples, it becomes easier for the model to memorize rather than generalize.

            - **Limitations of TF-IDF:**
            1. **Sparsity:**
                - TF-IDF matrices are sparse by nature and may still lead to computational inefficiency. While it provides better weighting, large feature spaces can still result in sparsity.
            2. **Handling of Rare Terms:**
                - If a term is extremely rare in the dataset (appearing only in one disease category), TF-IDF may give it low weight even if it is crucial for distinguishing that disease. Rare, but crucial terms may be ignored by TF-IDF if they appear in too few documents.
            3. **Dependency on Text Data Quality:**
                - The quality of the results highly depends on the quality of the textual data. If the disease descriptions, risk factors, or symptoms are inconsistent or poorly written, TF-IDF may not provide meaningful features. Additionally, it struggles with synonyms or phrases that convey the same meaning, such as “heart attack” and “myocardial infarction.”
            4. **Lack of Domain Knowledge Incorporation:**
                - While TF-IDF adjusts based on term frequency, it does not account for the clinical significance of terms unless it is combined with domain-specific preprocessing or feature engineering. For example, recognizing that a "stroke" is a major term across neurological diseases is not captured inherently by TF-IDF unless supported by domain knowledge.

            ### Conclusion:

            - **TF-IDF** appears to be a more sophisticated encoding method for disease-related text data, as it takes into account the importance of individual terms, leading to more informative and discriminative features. This allows the model to perform better, especially when distinguishing diseases based on their unique risk factors, symptoms, and signs. It is also more aligned with how diseases are clinically categorized and understood.
            
            - **One-Hot Encoding**, while simpler and more interpretable, may struggle with high-dimensionality and sparsity, limiting its ability to capture complex relationships within the data. It might be suitable for structured datasets with less variation and complexity but falls short when dealing with text data.

            By comparing the two methods, we can see that while both have their place, TF-IDF is generally more suited for medical text classification tasks where the relationships between terms and their importance are critical to disease categorization.
    """)
else:
    st.info("Please upload both 'disease_features.csv' and 'encoded_output2.csv' to continue.")