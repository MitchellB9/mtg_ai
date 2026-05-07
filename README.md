# MTG AI Project

## Quickstart
1. .\.venv\Scripts\Activate.ps1
2. Install requirements
3. Run pipeline:


TODO:
- ensure self-referential text (this__card) also accounts for partial name (Frodo, Sauron's Bane = Frodo)
- expand MULTIWORD_PHRASES in oracle_tokenizer.py_
- unignore src/models/__init__.py (conflict with Artifacts/Models?)
- create sub-datasets to utilize all data from bulk
	- card_text_tokens (cleaned oracle text)
	- card_features (Derived / ML-ready features)
- move reminder text to its own column
- 

Command Sequence:

python -m src.pipelines.build_dataset
python -m src.pipelines.preprocess_text
python -m src.pipelines.build_vectors
python -m src.pipelines.cluster_cards
python -m src.pipelines.find_similar
