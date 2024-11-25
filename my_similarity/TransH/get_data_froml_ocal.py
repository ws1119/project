import pandas as pd

entity2id_set = set()
relation2id_set = set()

df = pd.read_csv("all_triples.csv")

for index, row in df.iterrows():
    _h = row[0]
    entity2id_set.add(_h)
    _r = row[1]
    relation2id_set.add(_r)
    _t = row[2]
    entity2id_set.add(_t)
    with open("data/train.txt", "a", encoding="utf-8") as t:
        t.write(_h + "\t" + _r + "\t" + _t + "\n")


entity_id = 0
relation_id = 0
for entity in entity2id_set:
    with open("data/entity2id.txt", "a", encoding="utf-8") as h:
            h.write(entity + "\t" + str(entity_id) + "\n")
    entity_id += 1

# 写入关系以及对应id
for relation in relation2id_set:
    with open("data/relation2id.txt", "a", encoding="utf-8") as t:
        t.write(relation + "\t" + str(relation_id) + "\n")
        relation_id += 1

