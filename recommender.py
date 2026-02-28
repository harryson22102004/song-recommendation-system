import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SongRecommender:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['lyrics'])
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

    def recommend(self, song_title, top_n=5):
        if song_title not in self.df['title'].values:
            return ["Song not found in database"]

        idx = self.df[self.df['title'] == song_title].index[0]
        similarity_scores = list(enumerate(self.similarity_matrix[idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        recommendations = []
        for i in similarity_scores[1:top_n+1]:
            recommendations.append(self.df.iloc[i[0]]['title'])

        return recommendations
