from sklearn.feature_extraction.text import TfidfVectorizer

df['RECRUIT_ID'] = df['RECRUIT_ID'].astype(str)
df['PERSON_ID'] = df['PERSON_ID'].astype(str)

data = df.groupby('RECRUIT_ID')['PERSON_ID'].apply(list).reset_index()

tv = TfidfVectorizer()
tv_fit = tv.fit_transform(data['PERSON_ID'].astype(str))

word_list = tv.get_feature_names()
word_dict = tv.vocabulary_

#获取向量
word_emb = tv_fit.toarray()
