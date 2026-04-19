## Generating FAISS Index

1. Create the database

```
source .venv/bin/activate
python ast-script/extract_graph.py
```

2. Generate the index

```
deactivate
conda activate faiss
python ast-script/build_index.py
```