# Evaluation results

| lang   | mode   |   n |   accuracy |   hallucination_rate |   citation_accuracy |   refusal_accuracy_on_unanswerable |   retrieval_hit@3 |
|:-------|:-------|----:|-----------:|---------------------:|--------------------:|-----------------------------------:|------------------:|
| ar     | RAG    |  10 |      0.400 |                0.200 |               0.167 |                              1.000 |             0.667 |
| en     | RAG    |  10 |      0.700 |                0.200 |               0.333 |                              1.000 |             0.333 |
| ar     | no-RAG |  10 |      0.100 |                0.900 |               0.000 |                              0.000 |           nan     |
| en     | no-RAG |  10 |      0.200 |                0.800 |               0.000 |                              0.000 |           nan     |

accuracy / hallucination_rate / citation_accuracy are fractions of all questions in that group (n=10 per lang x mode, 20 total per mode, 40 overall). refusal_accuracy_on_unanswerable is restricted to the 4 unanswerable questions per lang. retrieval_hit@3 is restricted to answerable questions in RAG mode.