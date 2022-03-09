import gensim
from glove import Glove
from glove import Corpus

df['RECRUIT_ID'] = df['RECRUIT_ID'].astype(str)
df['PERSON_ID'] = df['PERSON_ID'].astype(str)

data = df.groupby('RECRUIT_ID')['PERSON_ID'].apply(list).reset_index()

sentences = data['PERSON_ID'].values.tolist()
all_words_vocabulary = []
for i in sentences: all_words_vocabulary.extend(i)

#建立语料输入模型
corpus_model = Corpus()
corpus_model.fit(sentences, window=10)
#返回单词字典，values为对应的索引
word_dict = corpus_model.dictionary

#glove模型训练
glove_model = Glove(no_components=16, learning_rate=0.05)
glove_model.fit(corpus_model.matrix, epochs=10, no_threads=1, verbose=True)
glove_model.add_dictionary(corpus_model.dictionary)

#获取每个单词的emb向量
emb_dict = {}
for word_i in all_words_vocabulary:
    if word_i in glove.dictionary:
        emb_dict[word_i] = glove.word_vectors[glove.dictionary[word_i]]
    else:
        emb_dict[word_i] = np.zeros(16, dtype="float32")  

#将向量进行平均
def build_vector(text,size,glove_model):
    vec = np.zeros(size).reshape((1,size))
    count = 0
    for w in text:   
        vec += glove_model.word_vectors[glove_model.dictionary[w]].reshape((1,size))
        count +=1
    if count!=0:
        vec/=count
    return vec
    
glove_emb = np.concatenate([build_vector(z, 16, glove_model) for z in sentences])
glove_emb = pd.DataFrame(glove_emb)
