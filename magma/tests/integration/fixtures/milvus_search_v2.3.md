# Milvus Collection.search Reference

Source: pymilvus.orm.collection.Collection.search docstring (pymilvus==3.0.0)

## Parameters

- data (``List[List[float]]/sparse types``): The vectors of search data.
the length of data is number of query (nq),
and the dim of every vector in data must be equal to the vector field of collection.
- anns_field (``str``): The name of the vector field used to search of collection.
- top_k (``int``): The max number of returned record, the maximum number of results to return.
