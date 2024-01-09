import time
import argparse
from tqdm import tqdm
from datasets import load_dataset
from setup_model_and_pipeline import get_aos_client

def deduplicate_dataset(dataset):
    context_list = [row["context"] for row in dataset]
    context_set = set(context_list)
    return list(context_set)

def build_bulk_body(index_name,sources_list):
    bulk_body = []
    for source in sources_list:
        bulk_body.append({ "index" : { "_index" : index_name} })
        bulk_body.append(source)
    return bulk_body

def ingest_dataset(dataset,aos_client,index_name, bulk_size=50):
    print("Deduplicating dataset...")
    context_list = deduplicate_dataset(dataset)
    # 19029 for train, 1204 for validation
    print(f"Finished deduplication. Total number of passages: {len(context_list)}")
    
    for start_idx in tqdm(range(0,len(context_list),bulk_size)):
        contexts = context_list[start_idx:start_idx+bulk_size]
        response = aos_client.bulk(
            build_bulk_body(index_name, [{"content":context} for context in contexts]),
            # set a large timeout because a new sparse encoding endpoint need warm up
            request_timeout=100
        )
        assert response["errors"]==False
    
    aos_client.indices.refresh(index=index_name,request_timeout=100)
    
def calculate_recall_rate(dataset, index_name, aos_client, data_size, query_body_lambda, recall_k=4):
    hit_cnt = 0
    miss_cnt = 0
    for idx, item in tqdm(enumerate(dataset.select(range(data_size)))):
        query = item['question']
        content = item['context']
        response = aos_client.search(index=index_name,size=recall_k, body=query_body_lambda(query))
        docs = [hit["_source"]['content'] for hit in response["hits"]["hits"]]
        if content in docs:
            hit_cnt += 1
        else:
            miss_cnt += 1
    print(f"hit:{hit_cnt}, miss:{miss_cnt}, recall@{recall_k}:{hit_cnt/data_size}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--aos_endpoint', type=str, default='', help='aos endpoint')
    parser.add_argument('--testset_size', type=int, default=1000, help='testset size')
    parser.add_argument('--index_name', type=str, default='', help='index name')
    parser.add_argument('--is_ingest', type=bool, default=False, help='ingest or search')
    parser.add_argument('--topk', type=int, default=4, help='top k')
    parser.add_argument('--dense_model_id', type=str, default='', help='dense_model_id')
    parser.add_argument('--sparse_model_id', type=str, default='', help='sparse_model_id')
    parser.add_argument("--ingest", action="store_true", help="is ingest or search")
    parser.add_argument("--query_dataset_type", type=str, default='validation', help='use validation set or train set to query')
    args = parser.parse_args()
    aos_endpoint = args.aos_endpoint
    aos_domain = '-'.join(aos_endpoint.split('-')[1:3])
    testset_size = args.testset_size
    index_name = args.index_name
    ingest = args.ingest
    topk = args.topk
    dense_model_id = args.dense_model_id
    sparse_model_id = args.sparse_model_id
    query_dataset_type = args.query_dataset_type

    dataset_name = "squad_v2"
    dataset = load_dataset(dataset_name)

    aos_client = get_aos_client(aos_endpoint)

    if ingest is True:
        start = time.time()
        ingest_dataset(dataset=dataset["train"],aos_client=aos_client,index_name=index_name)
        ingest_dataset(dataset=dataset["validation"],aos_client=aos_client,index_name=index_name)
        elpase_time = time.time() - start
        throughput = float(testset_size)/elpase_time
        print(f"[ingest] throughput/s:{throughput}")
    else:
        print("start search by bm25")
        calculate_recall_rate(
            dataset = dataset[query_dataset_type],
            index_name = index_name,
            aos_client = aos_client,
            data_size = testset_size,
            query_body_lambda = lambda query_text: {
                "query": {
                    "match": {
                        "content" : query_text
                    }
                }
            },
            recall_k=topk
        )
        
        print("start search by dense")
        calculate_recall_rate(
            dataset = dataset[query_dataset_type],
            index_name = index_name,
            aos_client = aos_client,
            data_size = testset_size,
            query_body_lambda = lambda query_text: {
                "query": {
                    "neural": {
                        "dense_embedding": {
                        "query_text": query_text,
                        "model_id": dense_model_id,
                        "k": topk
                        }
                    }
                }
            },
            recall_k=topk
        )
        
        print("start search by sparse")
        calculate_recall_rate(
            dataset = dataset[query_dataset_type],
            index_name = index_name,
            aos_client = aos_client,
            data_size = testset_size,
            query_body_lambda = lambda query_text: {
                "query": {
                    "neural_sparse": {
                    "sparse_embedding": {
                        "query_text": query_text,
                        "model_id": sparse_model_id,
                        "max_token_score": 3.5
                    }
                }
                }
            },
            recall_k=topk
        )
        
        print("start search by dense-sparse")
        calculate_recall_rate(
            dataset = dataset[query_dataset_type],
            index_name = index_name,
            aos_client = aos_client,
            data_size = testset_size,
            query_body_lambda = lambda query_text: {
                "query": {
                    "hybrid": {
                        "queries": [
                            {
                                "neural_sparse": {
                                    "sparse_embedding": {
                                        "query_text": query_text,
                                        "model_id": sparse_model_id,
                                        "max_token_score": 3.5
                                    }
                                }
                            },
                            {
                                "neural": {
                                    "dense_embedding": {
                                        "query_text": query_text,
                                        "model_id": dense_model_id,
                                        "k": 10
                                    }
                                }
                            }
                        ]
                    }
                }
            },
            recall_k=topk
        )
        
        print("start search by dense-bm25")
        calculate_recall_rate(
            dataset = dataset[query_dataset_type],
            index_name = index_name,
            aos_client = aos_client,
            data_size = testset_size,
            query_body_lambda = lambda query_text: {
                "query": {
                    "hybrid": {
                        "queries": [
                            {
                                "match": {
                                    "content" : query_text
                                }
                            },
                            {
                                "neural": {
                                    "dense_embedding": {
                                        "query_text": query_text,
                                        "model_id": dense_model_id,
                                        "k": 10
                                    }
                                }
                            }
                        ]
                    }
                }
            },
            recall_k=topk
        )
