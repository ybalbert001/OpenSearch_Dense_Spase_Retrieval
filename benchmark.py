from search_func import search_by_bm25, search_by_dense, search_by_sparse, search_by_dense_sparse, search_by_dense_bm25
from setup_model_and_pipeline import get_aos_client
from datasets import load_dataset
import time
import argparse
import json

def ingest_data(aos_client, index_name, content):    
    request_body = {
        "content": content
    }

    response = aos_client.transport.perform_request(
        method="POST",
        url=f"/{index_name}/_doc",
        body=json.dumps(request_body)
    )

    return response

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
    args = parser.parse_args()
    aos_endpoint = args.aos_endpoint
    aos_domain = '-'.join(aos_endpoint.split('-')[1:3])
    testset_size = args.testset_size
    index_name = args.index_name
    ingest = args.ingest
    topk = args.topk
    dense_model_id = args.dense_model_id
    sparse_model_id = args.sparse_model_id

    dataset_name = "squad_v2"
    dataset = load_dataset(dataset_name)

    aos_client = get_aos_client(aos_endpoint)

    if ingest is True:
        start = time.time()
        for idx, item in enumerate(dataset["train"].select(range(testset_size))):
            try:
                response = ingest_data(aos_client, index_name, item['context'][:2000])
                if idx % 50 == 0:
                    print(f"{idx}-th ingested.")
            except Exception as e:
                print(e)
                print(item['context'])
        elpase_time = time.time() - start
        throughput = float(testset_size)/elpase_time
        print(f"[ingest] throughput/s:{throughput}")
    else:
        hit_cnt = 0
        miss_cnt = 0
        start = time.time()
        for idx, item in enumerate(dataset["train"].select(range(testset_size))):
            query = item['question']
            content = item['context']
            if idx % 50 == 0:
                print(f"{idx}-th searched.")
            results = search_by_bm25(aos_client, index_name, query, topk)
            if content in results:
                hit_cnt += 1
            else:
                miss_cnt += 1

        recall_k = float(hit_cnt)/testset_size
        elpase_time = time.time() - start
        throughput = float(testset_size)/elpase_time
        print(f"[search_by_bm25] hit:{hit_cnt}, miss:{miss_cnt}, recall@{topk}: {recall_k}, throughput/s:{throughput}")

        hit_cnt = 0
        miss_cnt = 0
        start = time.time()
        for idx, item in enumerate(dataset["train"].select(range(testset_size))):
            query = item['question']
            content = item['context']
            if idx % 50 == 0:
                print(f"{idx}-th searched.")
            results = search_by_dense(aos_client, index_name, query, dense_model_id, topk)
            if content in results:
                hit_cnt += 1
            else:
                miss_cnt += 1

        recall_k = float(hit_cnt)/testset_size
        elpase_time = time.time() - start
        throughput = float(testset_size)/elpase_time
        print(f"[search_by_dense] hit:{hit_cnt}, miss:{miss_cnt}, recall@{topk}: {recall_k}, throughput/s:{throughput}")

        hit_cnt = 0
        miss_cnt = 0
        start = time.time()
        for idx, item in enumerate(dataset["train"].select(range(testset_size))):
            query = item['question']
            content = item['context']
            if idx % 50 == 0:
                print(f"{idx}-th searched.")
            results = search_by_sparse(aos_client, index_name, query, sparse_model_id, topk)
            if content in results:
                hit_cnt += 1
            else:
                miss_cnt += 1

        recall_k = float(hit_cnt)/testset_size
        elpase_time = time.time() - start
        throughput = float(testset_size)/elpase_time
        print(f"[search_by_sparse] hit:{hit_cnt}, miss:{miss_cnt}, recall@{topk}: {recall_k}, throughput/s:{throughput}")

        hit_cnt = 0
        miss_cnt = 0
        start = time.time()
        for idx, item in enumerate(dataset["train"].select(range(testset_size))):
            query = item['question']
            content = item['context']
            if idx % 50 == 0:
                print(f"{idx}-th searched.")
            results = search_by_dense_sparse(aos_client, index_name, query, sparse_model_id, dense_model_id, topk)
            if content in results:
                hit_cnt += 1
            else:
                miss_cnt += 1

        recall_k = float(hit_cnt)/testset_size
        elpase_time = time.time() - start
        throughput = float(testset_size)/elpase_time
        print(f"[search_by_dense_sparse] hit:{hit_cnt}, miss:{miss_cnt}, recall@{topk}: {recall_k}, throughput/s:{throughput}")

        hit_cnt = 0
        miss_cnt = 0
        start = time.time()
        for idx, item in enumerate(dataset["train"].select(range(testset_size))):
            query = item['question']
            content = item['context']
            if idx % 50 == 0:
                print(f"{idx}-th searched.")
            results = search_by_dense_bm25(aos_client, index_name, query, dense_model_id, topk)
            if content in results:
                hit_cnt += 1
            else:
                miss_cnt += 1

        recall_k = float(hit_cnt)/testset_size
        elpase_time = time.time() - start
        throughput = float(testset_size)/elpase_time
        print(f"[search_by_dense_bm25] hit:{hit_cnt}, miss:{miss_cnt}, recall@{topk}: {recall_k}, throughput/s:{throughput}")
