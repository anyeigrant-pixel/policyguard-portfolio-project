import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
from src.utils.config import DATA_DIR, ARTIFACT_DIR

def main():
    df = pd.read_csv(DATA_DIR/'processed'/'policyholders.csv')
    cv = CountVectorizer(stop_words='english', ngram_range=(1,2), min_df=5, max_features=1000)
    X = cv.fit_transform(df['service_note'])
    counts = X.sum(axis=0).A1
    terms = pd.DataFrame({'term':cv.get_feature_names_out(),'count':counts}).sort_values('count', ascending=False).head(50)
    terms.to_csv(ARTIFACT_DIR/'top_terms.csv', index=False)

    tf = TfidfVectorizer(stop_words='english', min_df=5, max_features=2000)
    Xt = tf.fit_transform(df['service_note'])
    nmf = NMF(n_components=5, random_state=42)
    W = nmf.fit_transform(Xt)
    feats = tf.get_feature_names_out()
    topics=[]
    for i, comp in enumerate(nmf.components_):
        top = [feats[j] for j in comp.argsort()[-8:][::-1]]
        topics.append({'topic':i+1,'top_terms':', '.join(top)})
    pd.DataFrame(topics).to_csv(ARTIFACT_DIR/'topics.csv', index=False)
    df[['customer_id','service_note']].assign(topic=W.argmax(axis=1)+1).to_csv(ARTIFACT_DIR/'note_topics.csv', index=False)
    print('text analytics artifacts written')

if __name__ == '__main__':
    main()
