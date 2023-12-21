# OpenSearch_Dense_Spase_Retrieval

- Check the Dataset
    refer sparse_dense_squad_v2.ipynb
- Setup OpenSearch Index with Dense and Sparse vectors
- Setup pipelines for ingestion and searching
    ```shell
    python3 setup_model_and_pipeline.py --aos_endpoint {aos_endpoint} --sparse_model_id {sparse_model_id} --index_name {index_name}
    ```
- Test Data Ingestion
    ```shell
    # ingest data
    python3 benchmark.py --aos_endpoint "vpc-domain66ac69e0-2m4jji7cweof-4fefsofiqdzu3hxammxwq5hth4.us-west-2.es.amazonaws.com" --testset_size 3000 --index_name "aos-retrieval" --topk 4 --dense_model_id "F1Nhh4wBpwn7Z6ncnUsT" --sparse_model_id "EVNZh4wBpwn7Z6ncaEtm" --ingest
    ```
- Benchmarking
    ```shell
    python3 benchmark.py --aos_endpoint "vpc-domain66ac69e0-2m4jji7cweof-4fefsofiqdzu3hxammxwq5hth4.us-west-2.es.amazonaws.com" --testset_size 1000 --index_name "aos-retrieval" --topk 4 --dense_model_id "GLthh4wBsY2vwfNenYNZ" --sparse_model_id "EVNZh4wBpwn7Z6ncaEtm"
    ```

