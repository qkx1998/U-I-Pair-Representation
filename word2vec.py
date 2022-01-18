from gensim.models import Word2Vec

df['RECRUIT_ID'] = df['RECRUIT_ID'].astype(str)
df['PERSON_ID'] = df['PERSON_ID'].astype(str)

data = df.groupby('RECRUIT_ID')['PERSON_ID'].apply(list).reset_index()
sequence = data['PERSON_ID'].values.tolist()
model = Word2Vec(sequence, size=16, window=10, min_count=0, seed=2, workers=6, sg=1, iter=10)
#获取单词列表
model.wv.index2word

#获取每个单词对应的向量
w2v_vec_dict = {}
for word in model.wv.index2word:
    w2v_vec_dict[word] = list(model[word])

#将序列中的单词向量进行平均
def build_vector(text,size,wv):
    vec = np.zeros(size).reshape((1,size))
    count = 0
    for w in text:   
        vec +=  wv[w].reshape((1,size))
        count +=1
    if count!=0:
        vec/=count
    return vec
    
w2v_emb = np.concatenate([build_vector(z, 16, model.wv) for z in sequence])
w2v_emb = pd.DataFrame(w2v_emb)
