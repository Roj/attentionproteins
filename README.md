## Attention, proteins!


Installation instructions:

```bash
git clone https://github.com/Roj/attentionproteins.git && cd attentionproteins
python3.8 -m venv env3
. env3/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pre-commit install # for development
```

To run the main model: `python src/GATPipeline.py` (make sure you have the dataset specified in
`GRAPHS_PKL`!)
