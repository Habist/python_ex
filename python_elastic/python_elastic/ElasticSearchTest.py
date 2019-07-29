from elasticsearch import Elasticsearch

#Elastic Setting
es = Elasticsearch(
    ['192.168.35.202']
)

#인덱스 생성
# es.indices.create(index='test_create',
#                   body={
#                           "settings": {
#                             "number_of_shards": 1
#                           },
#                           "mappings": {
#                             "properties": {
#                               "field1":{
#                                 "type":"keyword"
#                               }
#                             }
#                           }
#                         })

#데이터 삽입
es.index(index='test_create',
         body={
  "field1":"33333"
})