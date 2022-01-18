from collections import defaultdict
from gensim.models import Word2Vec
import random
df_ = df[['RECRUIT_ID','PERSON_ID']].head(10000)
df_ = df_.astype(str)

# 参考https://github.com/zhangqibot/Tencent2020_Top5/tree/master/get_emb
# 构建U-I关系图
dic = defaultdict(set)
for item in df_.values:
    dic['PERSON_ID_' + item[1]].add('RECRUIT_ID_' + item[0])
    dic['RECRUIT_ID_' + item[0]].add('PERSON_ID_' + item[1])
dic_cont = {}
for key in dic:
    dic[key] = list(dic[key])
    dic_cont[key] = len(dic[key])    
    
# 构建路径
sentences = []
length = []
for key in dic:
    sentence = [key]
    while len(sentence) != path_length:
        rdm_index = random.randint(0, dic_cont[sentence[-1]] - 1)
        key = dic[sentence[-1]][rdm_index]
        if len(sentence) >= 2 and key == sentence[-2]:
            break
        else:
            sentence.append(key)
    sentences.append(sentence)
    length.append(len(sentence))
    if len(sentences) % 100000 == 0:
        print(len(sentences))
        
random.shuffle(sentences)
model = Word2Vec(sentences, size=16, window=10, workers=6, min_count=1, iter=10, seed=2)

#得到每个单词对应的向量
all_words_vocabulary = set(df_['PERSON_ID'].values)
emb_dict = {}  
for word_i in all_words_vocabulary:
    emb_dict[word_i] = model.wv[f'PERSON_ID_{word_i}'] 

#将向量进行平均
def build_vector(text,size,wv):
    vec = np.zeros(size).reshape((1,size))
    count = 0
    for w in text:   
        vec +=  wv[w].reshape((1,size))
        count +=1
    if count!=0:
        vec/=count
    return vec

dw_emb = np.concatenate([build_vector(z, 16, model.wv) for z in sentences])
dw_emb = pd.DataFrame(dw_emb)
