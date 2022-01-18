from sklearn.feature_extraction.text import CountVectorizer

df['RECRUIT_ID'] = df['RECRUIT_ID'].astype(str)
df['PERSON_ID'] = df['PERSON_ID'].astype(str)

data = df.groupby('RECRUIT_ID')['PERSON_ID'].apply(list).reset_index()

cv = CountVectorizer()
cv_fit = cv.fit_transform(data['PERSON_ID'].astype(str))
#获取单词列表
word_list = cv.get_feature_names()
#获取单词字典，values为单词对应位置
word_dict = cv.vocabulary_
#获取向量
word_emb = cv_fit.toarray()
