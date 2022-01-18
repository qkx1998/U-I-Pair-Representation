from gensim.models.doc2vec import Doc2Vec, TaggedDocument

df['RECRUIT_ID'] = df['RECRUIT_ID'].astype(str)
df['PERSON_ID'] = df['PERSON_ID'].astype(str)

data = df.groupby('RECRUIT_ID')['PERSON_ID'].apply(list).reset_index()

sequence = [TaggedDocument(words=wdi[1], tags=[wdi[0]]) for wdi in data[['RECRUIT_ID', 'PERSON_ID']].values]
model = Doc2Vec(sequence, vector_size=16 , window=10 ,min_count=0, seed=2, workers=6, epoch=10)
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
    
sequence_list = [x[0] for x in sequence]
d2v_emb = np.concatenate([build_vector(z, 16, model.wv) for z in sequence_list])
d2v_emb = pd.DataFrame(d2v_emb)
