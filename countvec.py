from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
cv_fit = cv.fit_transform(data['PERSON_ID'].astype(str))
#获取单词列表
word_list = cv.get_feature_names()
#获取单词字典，values为单词对应位置
word_dict = cv.vocabulary_
#获取向量
word_emb = cv_fit.toarray()
