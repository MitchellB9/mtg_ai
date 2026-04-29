# MTG AI Project

## Quickstart
1. activate venv .\.venv\Scripts\Activate.ps1
2. Install requirements
3. Run pipeline:


TODO:
- have build_dataset.py replace old (Raw) dataset
- ensure self-referential text (this__card) also accounts for partial name (Frodo, Sauron's Bane = Frodo)
- expand MULTIWORD_PHRASES in oracle_tokenizer.py_
- unignore src/models/__init__.py (conflict with Artifacts/Models?)
- create sub-datasets to utilize all data from bulk
	- cards_main (One row per unique card, Only gameplay-relevant + commonly used fields)
	- card_prints (One row per printing)
	- card_text_tokens (cleaned oracle text)
	- card_features (Derived / ML-ready features)
	- card_ids (All ID relationships)
	- cards_misc (cols w/ mostly null values, inconsistent structure, niche use cases)
	- non_card_entities (tokens, vanguard, art_series,emblems, planar, scheme)
- move reminder text to its own column
- add vizualizations
- remove non-cards from dataset (tokens, vanguard, art_series,emblems, planar, scheme)
- 

Command Sequence:

python -m src.pipelines.build_dataset
python -m src.pipelines.preprocess_text
python -m src.pipelines.build_vectors
python -m src.pipelines.cluster_cards
python -m src.pipelines.find_similar
