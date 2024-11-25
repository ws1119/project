from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config

# define a config
config = Config()
config.max_connection_pool_size = 10
# init connection pool
connection_pool = ConnectionPool()
# if the given servers are ok, return true, else return false
ok = connection_pool.init([('127.0.0.1', 9669)], config)

# option 1 control the connection release yourself
# get session from the pool
session = connection_pool.get_session('root', 'nebula')

# select space
session.execute('USE remote_data2')

# show tags
# result = session.execute('SHOW edges')
# print(result)

# print(session.execute('GET SUBGRAPH 1 steps FROM "道路准确提取" yield vertices as node, edges as relationships'))
result_node = session.execute("match (r) return r")
# print(result)
# 获取图谱中的节点
get_result = []
for i in result_node:
    for j in i:
        re = str(j)[2:str(j).index('" :')]
        get_result.append(re)
for i in get_result:
    pass
    # print(i)

node_relation_result = session.execute("match()-[r]->() return r")


entity2id_set = set()
relation2id_set = set()

for i in node_relation_result:
    print("triple===",i)
    str_result = str(i)
    _h = str_result[2:str_result.find("[") - 3]
    _h = _h.strip()
    entity2id_set.add(_h)
    print("head entity===", _h)
    _t = str_result[str_result.find(">") + 3:-2]
    _t = _t.strip()
    entity2id_set.add(_t)
    print("tail entity===", _t)
    _r = str_result[str_result.find(":") + 1 : str_result.find("@")]
    relation2id_set.add(_r)
    # 写入三元组
    with open("../TransH/data/train.txt", "a", encoding="utf-8") as t:
        t.write(_h + "\t" + _r + "\t" + _t + "\n")
    print("relation===", _r)


entity_id = 0
relation_id = 0
for entity in entity2id_set:
    with open("../TransH/data/entity2id.txt", "a", encoding="utf-8") as h:
            h.write(entity + "\t" + str(entity_id) + "\n")
    entity_id += 1

# 写入关系以及对应id
for relation in relation2id_set:
    with open("../TransH/data/relation2id.txt", "a", encoding="utf-8") as t:
        t.write(relation + "\t" + str(relation_id) + "\n")
        relation_id += 1

# print(node_relation_result)

# release session
session.release()

# option 2 with session_context, session will be released automatically
# with connection_pool.session_context('root', 'nebula') as session:
#     session.execute('USE nba')
#     result = session.execute('SHOW TAGS')
#     print(result)

# close the pool
connection_pool.close()