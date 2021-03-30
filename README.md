## Attention, proteins!


Installation instructions:

```bash
git clone https://github.com/Roj/attentionproteins.git && cd attentionproteins
conda env create -f conda.yaml
conda activate attention_proteins
pre-commit install # for development
```

To run the main model: `python src/GATPipeline.py` (make sure you have the dataset specified in
`GRAPHS_PKL`!)

Alternatively, you can just do `mlflow run .` to trigget the pipeline, with all dependencies
encapsulated. This requires mlflow to work.  To view the results in a nice format,
run `mlflow ui` in the terminal.