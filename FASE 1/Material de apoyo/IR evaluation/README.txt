IR Evaluation Sample Files

This directory contains a small, independent example of the input files used
by the information-retrieval evaluator. The sample files document the
expected tabular format and make it possible to verify the evaluator without
using the complete project corpus.

The number of relevant documents may vary across the 15 queries.
Counts range from 10 to 15, matching the assignment specification. 

qrels_sample.tsv: reference relevance judgments for each query and document.
It contains the columns query_id, document_id, and relevance.
rankings_sample.tsv: retrieved-document rankings to be evaluated. It contains
the columns query_id, document_id, rank, and score.

The relevance scale is 1 for partially relevant and 2 for highly relevant.
Documents absent from qrels are treated as having relevance 0 for this
exercise.

The evaluator can be run with:
python evaluate_ir_metrics.py rankings_sample.tsv qrels_sample.tsv
