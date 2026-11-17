---
license: cc-by-2.5
task_categories:
- question-answering
- sentence-similarity
language:
- en
tags:
- rag
- dpr
- information-retrieval
- question-answering
- biomedical
configs:
- config_name: text-corpus
  data_files: 
  - split: passages
    path: "data/passages.parquet/*"
- config_name: question-answer-passages
  data_files: 
  - split: test
    path: "data/test.parquet/*"
---

See [here](https://huggingface.co/datasets/enelpol/rag-mini-bioasq) for an updated version without nans in text-corpus.

[In this huggingface discussion](https://discuss.huggingface.co/t/what-are-you-using-the-mini-bioasq-dataset-for/89042?u=tillwenke) you can share what you used the dataset for.

Derives from http://participants-area.bioasq.org/Tasks/11b/trainingDataset/ we generated our own subset using `generate.py`.
