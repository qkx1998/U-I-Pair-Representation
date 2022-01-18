import networkx as nx
from collections import Counter
from node2vec import Node2Vec

# 参考https://www.kaggle.com/ferdzso/knowledge-graph-analysis-with-node2vec
def add_nodes(G, df, col, type_name):
    """Add entities to G from the 'col' column of the 'df' DataFrame. The new nodes are annotated with 'type_name' label."""
    nodes = list(df[~df[col].isnull()][col].unique())
    G.add_nodes_from([(n,dict(type=type_name)) for n in nodes])
    print("Nodes (%s,%s) were added" % (col, type_name))
    
def add_links(G, df, col1, col2, type_name):
    """Add links to G from the 'df' DataFrame. The new edges are annotated with 'type_name' label."""
    df_tmp = df[(~df[col1].isnull()) & (~df[col2].isnull())]
    links = list(zip(df_tmp[col1],df_tmp[col2]))
    G.add_edges_from([(src, trg, dict(type=type_name)) for src, trg in links])
    print("Edges (%s->%s,%s) were added" % (col1, col2, type_name))
    
G = nx.DiGraph()
add_nodes(G, df_, "RECRUIT_ID", "RECRUIT_ID")
add_nodes(G, df_, "PERSON_ID", "PERSON_ID")
add_links(G, df_, "RECRUIT_ID", "PERSON_ID", "RECRUIT_ID_PERSON_ID")

#查看节点关系
G.in_edges('6256839')

#移除孤立的节点
print(G.number_of_nodes(), G.number_of_edges())
G.remove_nodes_from(list(nx.isolates(G)))
print(G.number_of_nodes(), G.number_of_edges())

#将节点编码为整数标识符
def encode_graph(G):
    """Encode the nodes of the network into integers"""
    nodes = [(n,d.get("type",None)) for n, d in G.nodes(data=True)]
    nodes_df = pd.DataFrame(nodes, columns=["id","type"]).reset_index()
    node2idx = dict(zip(nodes_df["id"],nodes_df["index"]))
    edges = [(node2idx[src], node2idx[trg], d.get("type",None)) for src, trg, d in G.edges(data=True)]
    edges_df = pd.DataFrame(edges, columns=["src","trg","type"])
    return nodes_df, edges_df

nodes_df, edges_df = encode_graph(G)
print(len(nodes_df), len(edges_df))

edge_list = list(zip(edges_df["src"],edges_df["trg"]))
KG = nx.Graph(edge_list)
print(KG.number_of_nodes(), KG.number_of_edges())

#训练node2vec模型
n2v_obj = Node2Vec(KG, dimensions=10, walk_length=5, num_walks=10, p=1, q=1, workers=1)
n2v_model = n2v_obj.fit(window=3, min_count=1, batch_words=4)

#获取embedding
def get_embeddings(model, nodes):
    """Extract representations from the node2vec model"""
    embeddings = [list(model.wv.get_vector(n)) for n in nodes]
    embeddings = np.array(embeddings)
    print(embeddings.shape)
    return embeddings
    
RECRUIT_ID = list(nodes_df[nodes_df["type"] == "RECRUIT_ID"]["index"])
PERSON_ID = list(nodes_df[nodes_df["type"] == "PERSON_ID"]["index"])
print(len(RECRUIT_ID), len(PERSON_ID))

RECRUIT_ID = list(set(RECRUIT_ID).intersection(set(KG.nodes())))
PERSON_ID = list(set(PERSON_ID).intersection(set(KG.nodes())))
print(len(RECRUIT_ID), len(PERSON_ID))

RECRUIT_ID = [str(item) for item in RECRUIT_ID]
PERSON_ID = [str(item) for item in PERSON_ID]

recruit_id_emb = get_embeddings(n2v_model, RECRUIT_ID)
person_id_emb = get_embeddings(n2v_model, PERSON_ID)

recruit_id_emb = pd.DataFrame(recruit_id_emb)
recruit_id_emb['RECRUIT_ID'] = RECRUIT_ID

person_id_emb = pd.DataFrame(person_id_emb)
person_id_emb['PERSON_ID'] = PERSON_ID
