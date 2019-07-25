from elasticsearch import Elasticsearch

es = Elasticsearch(
    ['192.168.35.202']
)

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


# es.index(index='test_create',
#          body=)