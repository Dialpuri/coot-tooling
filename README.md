## Generating FAISS Index

1. Create the database

```
source .venv/bin/activate
python ast-script/extract_graph.py
```

2. Generate the comments

```
python ast-script/summarise_functions.py
```

3. Generate the index

```
deactivate
conda activate faiss
python ast-script/build_index.py
```

4. Query a function

```
python ast-script/query.py
```

5. Create an oracle

```
python -m oracle.render coot::
```