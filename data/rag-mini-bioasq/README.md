# RAG Mini BioASQ

Local copy of https://huggingface.co/datasets/rag-datasets/rag-mini-bioasq.
The exact source revision, file sizes, and SHA-256 checksums are in
`manifest.json`. Both downloaded Parquet files were checked against the
publisher's checksums and successfully read with PyArrow.

| Configuration / split | File | Rows | Columns |
| --- | --- | ---: | --- |
| text-corpus / passages | `data/passages.parquet/part.0.parquet` | 40,221 | passage, id |
| question-answer-passages / test | `data/test.parquet/part.0.parquet` | 4,719 | question, answer, relevant_passage_ids, id |

The upstream dataset card and its license declaration (CC BY 2.5) are
preserved in `DATASET_CARD.md`. The question set is provided as a test split;
there is no separate training split in this download.

Load from the project root with pandas and a Parquet engine installed
(`uv pip install pandas pyarrow` in your activated environment):

```python
from pathlib import Path
import pandas as pd

root = Path("data/rag-mini-bioasq")
passages = pd.read_parquet(root / "data/passages.parquet/part.0.parquet")
questions = pd.read_parquet(root / "data/test.parquet/part.0.parquet")
```
