import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, silhouette_samples
from scipy.sparse import hstack
import plotly.express as px

st.set_page_config(layout="wide")
st.title("Advanced Job Analytics Dashboard")

# ==============================
# 1. Load dataset
# ==============================
df = pd.read_csv("cleanedmyjob.csv")
st.subheader("Dataset Overview")
st.markdown("""
This dataset contains job postings with features such as:  
- `title`: Job title  
- `salary`: Salary offered  
- `location`: Job location  
- `date_posted` & `closing_date`: For job duration calculations  
""")
st.dataframe(df.head())

# ==============================
# 2. Feature Engineering for Clustering
# ==============================
st.subheader("Feature Engineering for Clustering")
st.markdown("We transform job titles using TF-IDF and locations using one-hot encoding to prepare features for clustering.")

# Preprocess titles
stopwords_extra = ['senior','junior','developer','engineer','assistant','manager','consultant','limited','ltd']
tfidf = TfidfVectorizer(stop_words='english', max_features=2000, ngram_range=(1,2))
X_title = tfidf.fit_transform(df['title'].astype(str).apply(lambda x: ' '.join([w for w in x.split() if w not in stopwords_extra])))

# One-hot encode locations
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_loc = ohe.fit_transform(df[['location']])
X_loc_scaled = StandardScaler(with_mean=False).fit_transform(X_loc)

# Combine features
X = hstack([X_title, X_loc_scaled])
st.write(f"Combined feature matrix: {X.shape} (TF-IDF + Location features)")

# ==============================
# 3. KMeans Clustering
# ==============================
st.subheader("KMeans Clustering")
n_clusters = st.slider("Select number of clusters", 2, 10, 5)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

st.markdown("KMeans clusters jobs based on similarity of title keywords and location.")

# ==============================
# 4. Clustering Evaluation Metrics
# ==============================
st.subheader("Cluster Quality Metrics")
sil_score = silhouette_score(X, df['cluster'])
ch_score = calinski_harabasz_score(X.toarray(), df['cluster'])
db_score = davies_bouldin_score(X.toarray(), df['cluster'])

st.write(f"**Silhouette Score:** {sil_score:.3f} — how well jobs are grouped; higher is better.")
st.write(f"**Calinski-Harabasz Index:** {ch_score:.2f} — ratio of between-cluster variance to within-cluster variance; higher is better.")
st.write(f"**Davies-Bouldin Index:** {db_score:.3f} — average similarity between clusters; lower is better.")

# ==============================
# 5. Silhouette per Cluster Visualization
# ==============================
sil_values = silhouette_samples(X, df['cluster'])
df['silhouette'] = sil_values
fig_sil = px.box(df, x='cluster', y='silhouette', color='cluster', title="Silhouette Values per Cluster")
st.plotly_chart(fig_sil)

# ==============================
# 6. Salary Insights by Location
# ==============================
st.subheader("Salary Insights by Location")
loc_stats = df.groupby('location')['salary'].agg(['min','mean','max','count']).reset_index()
fig_loc = px.bar(loc_stats, x='location', y='mean', color='location', text='mean',
                 title="Average Salary by Location",
                 hover_data={'min':True,'mean':True,'max':True,'count':True,'location':False})
st.plotly_chart(fig_loc)
st.markdown("This graph shows which locations offer higher average salaries and how many jobs are in each location.")

# ==============================
# 7. Salary Distribution
# ==============================
st.subheader("Salary Distribution Across Jobs")
fig_salary = px.histogram(df, x='salary', nbins=20, title="Salary Distribution", marginal="box")
st.plotly_chart(fig_salary)
st.markdown("This histogram shows the frequency of different salary ranges. The boxplot on top helps identify outliers.")

# ==============================
# 8. Cluster Salary Analysis
# ==============================
st.subheader("Salary Statistics by Cluster")
cluster_stats = df.groupby('cluster')['salary'].agg(['min','mean','max','count']).reset_index()
fig_cluster = px.bar(cluster_stats, x='cluster', y='mean', color='cluster', text='mean',
                     title="Average Salary per Cluster",
                     hover_data={'min':True,'mean':True,'max':True,'count':True})
st.plotly_chart(fig_cluster)
st.markdown("This shows which clusters (groups of similar jobs) have higher average salaries.")

# ==============================
# 9. Top Keywords per Cluster
# ==============================
st.subheader("Top Job Title Keywords per Cluster")
terms = np.array(tfidf.get_feature_names_out())
for i in range(n_clusters):
    cluster_center = kmeans.cluster_centers_[:,:X_title.shape[1]][i]
    top_terms = [terms[j] for j in cluster_center.argsort()[-5:][::-1]]
    st.write(f"Cluster {i}: {', '.join(top_terms)}")

# ==============================
# 10. Monthly Salary Trends
# ==============================
if 'posted_month' in df.columns:
    st.subheader("Average Salary by Posting Month")
    month_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",
                 8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    df['posted_month_name'] = df['posted_month'].map(month_map)
    month_stats = df.groupby('posted_month_name')['salary'].agg(['min','mean','max','count']).reindex(
        ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]).reset_index()
    fig_month = px.bar(month_stats, x='posted_month_name', y='mean', text='mean',
                       title="Average Salary by Posting Month",
                       hover_data={'min':True,'mean':True,'max':True,'count':True,'posted_month_name':False})
    st.plotly_chart(fig_month)
    st.markdown("Shows seasonal salary trends — which months have higher paying jobs.")




# streamlit run models.py
