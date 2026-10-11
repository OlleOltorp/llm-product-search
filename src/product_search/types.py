"""Standard tables shared by data loading, retrieval and evaluation.

Plain dicts in the format that pytrec_eval and ir_measures expect.
"""

Products = dict[str, str]  # product_id -> text
Queries = dict[str, str]  # query_id -> text
Labels = dict[str, dict[str, int]]  # query_id -> {product_id: grade}
Results = dict[str, dict[str, float]]  # query_id -> {product_id: score}
