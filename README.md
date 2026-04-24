# compacting-bill-simpler

Nouveau point de départ pour reconstruire proprement la pipeline d'analyse des bills US LegiScan, en conservant les briques qui marchaient dans `Documents/compacting_bills`.

## Ce qui est repris tel quel depuis l'ancien projet

- nettoyage du texte législatif dans [src/compacting_bill_simpler/text_processing.py](/Users/alexandreperverie/Documents/compacting_bill_simpler/src/compacting_bill_simpler/text_processing.py)
- segmentation en phrases et réparation des phrases trop longues
- chunking par budget de tokens
- config de pipeline, presets de modèles, adaptation des kwargs OpenAI
- chargement du dataset compressé `.csv.gz`

Le but de ce repo n'est pas de garder toute l'ancienne suite tout de suite. Pour l'instant, la pipeline active s'arrête volontairement à :

`ingest -> clean -> segment -> chunk`

## Structure

- [src/compacting_bill_simpler/text_processing.py](/Users/alexandreperverie/Documents/compacting_bill_simpler/src/compacting_bill_simpler/text_processing.py) : nettoyage, token count, segmentation bas niveau, chunking brut
- [src/compacting_bill_simpler/input_files.py](/Users/alexandreperverie/Documents/compacting_bill_simpler/src/compacting_bill_simpler/input_files.py) : résolution `.csv` / `.csv.gz`
- [src/compacting_bill_simpler/regulatory/config.py](/Users/alexandreperverie/Documents/compacting_bill_simpler/src/compacting_bill_simpler/regulatory/config.py) : arguments/config pipeline
- [src/compacting_bill_simpler/regulatory/llm_profiles.py](/Users/alexandreperverie/Documents/compacting_bill_simpler/src/compacting_bill_simpler/regulatory/llm_profiles.py) : presets et kwargs OpenAI
- [src/compacting_bill_simpler/regulatory/stages/](/Users/alexandreperverie/Documents/compacting_bill_simpler/src/compacting_bill_simpler/regulatory/stages) : `ingest`, `segment`, `chunk`, `document_signals`
- [src/compacting_bill_simpler/regulatory/orchestrator.py](/Users/alexandreperverie/Documents/compacting_bill_simpler/src/compacting_bill_simpler/regulatory/orchestrator.py) : pipeline courte
- [src/compacting_bill_simpler/regulatory/cli.py](/Users/alexandreperverie/Documents/compacting_bill_simpler/src/compacting_bill_simpler/regulatory/cli.py) : CLI

## Commandes utiles

Après `uv sync` :

```bash
uv run compacting-bill-simpler --limit 1
uv run compacting-bill-simpler --bill-id ResInv --write-json
uv run python -m compacting_bill_simpler.regulatory --bill-id ResInv --show-cleaned-text
uv run compacting-bill-simpler --trace-bill-id 0021
```

## Sortie actuelle

La CLI affiche un aperçu JSON avec :

- métadonnées de la bill
- modèle/preset choisis
- texte nettoyé optionnel
- premières phrases segmentées
- premiers chunks
- signaux documentaires globaux

## Trace

Tu peux maintenant lancer un mode trace qui écrit un dossier `dataset/processed/trace_vN/` avec :

- `00_manifest.json`
- `01_raw_text.txt`
- `01_bill.json`
- `02_cleaned_text.txt`
- `03_sentences.jsonl`
- `04_chunks.jsonl`
- `05_document_profile.json`
- `06_preview.json`

Commande type :

```bash
uv run compacting-bill-simpler --trace-bill-id 0021
```

Et si tu veux imposer le dossier :

```bash
uv run compacting-bill-simpler --trace-bill-id 0021 --trace-dir dataset/processed/trace_v123
```
