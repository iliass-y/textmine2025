# TextMine 2025 relation extraction

Code of my entry to the TextMine 2025 challenge, held with the TextMine workshop of the EGC
conference in Strasbourg in January 2025. The task is relation extraction from French text: each
text comes with its annotated entities, and the system predicts the relations between them.

The method is described in the paper
[Tackling Class Imbalance in Relation Extraction for french text: Effective Negative Sampling and
Advanced Loss Functions](https://hal.science/hal-05099002)
(I. Ayaou, TextMine workshop, EGC 2025).

## 1. Method

The task is treated as multi-label classification of entity pairs.

- Negative sampling: pairs without a relation are labelled `no_relation`, up to three for each
  positive pair of a text, drawn from entity pairs whose types admit a relation (`negative_ratio=3`
  in `train.py`).
- Special tokens mark the entities and the candidate relations in the text. A self-reflective marker
  handles relations from an entity to itself, such as `GENDER_MALE` and `GENDER_FEMALE`, for the
  types `CIVILIAN`, `TERRORIST_OR_CRIMINAL` and `MILITARY`.
- Class weights follow the Effective Number of Samples method (`beta=0.9999`), with the smallest
  weight given to `no_relation`.
- Loss: asymmetric loss (`gamma_neg=4`, `gamma_pos=1`, `clip=0.05`). A focal loss trainer is kept
  in `train.py` as an alternative.
- One decision threshold per class, chosen on the validation split between 0.1 and 0.9.
- Base model in `train.py`: `cmarkea/distilcamembert-base`. The paper also reports CamemBERT base,
  which reached the highest macro F1.

## 2. Files

| File | Content |
|---|---|
| `train.py` | Data preparation, an 80/20 multilabel stratified split of the entity pairs built from `train.csv`, fine-tuning and threshold search |
| `eval.py` | Prediction on the test file and writing of `submission.csv` |
| `train.csv` | Training texts with entities and relations, 800 rows |
| `test_01-07-2024.csv` | Test texts with entities, 400 rows |

## 3. Requirements

Python with `pandas`, `numpy`, `torch`, `transformers` 4.41 to 4.45, `accelerate`, `scikit-learn`,
`iterative-stratification`, `spacy` and `tqdm`, and the spaCy model `fr_core_news_lg`
(`python -m spacy download fr_core_news_lg`). `train.py` uses `eval_strategy`, added in transformers
4.41, and overrides `compute_loss` without the `num_items_in_batch` argument that transformers 4.46
passes. Training uses `fp16=True` and 48 data loader workers, so it expects a CUDA GPU. The
experiments of the paper ran on one Tesla V100 with 32 GB.

## 4. Usage

1. `python train.py`. It writes `mlb.json`, `output_relations/unique_relations.json`,
   `best_thresholds.npy`, checkpoints under `results/`, and the fine-tuned model under
   `cmarkea/distilcamembert-base-finetuned/`.
2. In `eval.py`, set `base_model` to the path of the fine-tuned model. `eval.py` reads the
   thresholds from `optimal_tresholds.npy`, so copy `best_thresholds.npy` to that name first.
3. `python eval.py`. It writes `submission.csv`, with one JSON list of
   `[head_id, relation, tail_id]` triples per text.
