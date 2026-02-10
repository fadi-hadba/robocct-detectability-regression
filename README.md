# RoboCT Detectability Index Regression

Dieses Repository enthält Code zur **Vorhersage eines projektionsabhängigen Erkennbarkeitsindex** (Regression) aus industriellen CT‑Projektionen mittels Deep Learning (PyTorch Lightning + Hydra + Weights & Biases).

> **Hinweis zu Daten:** Rohdaten (z. B. `*.raw`, `*.npy`) sind **nicht** enthalten. Das Repo ist so aufgebaut, dass Experimente über YAML‑Configs reproduzierbar konfiguriert werden können.

## Features
- Regression: **Projection → Detectability/Erkennbarkeitsindex**
- Architekturen: **ResNet / VGG / EfficientNet** (`Network_Architectures/`)
- Experiment‑Konfiguration über **Hydra** (`config/`)
- Logging & Sweeps über **Weights & Biases (W&B)** (`config/sweep.yaml`)
- (Optional/WIP) ROI‑Fokus über Projektionsgeometrie (Codepfade sind vorhanden, teils auskommentiert)

## Projektstruktur
- `trainer.py` – Trainings‑Entry‑Point (Hydra + Lightning Trainer)
- `tester.py` – Test/Inference‑Skript (aktuell mit exemplarischen Pfaden)
- `lightning_network_methods.py` – LightningModule (Model/Loss/Optimizer/Training Steps)
- `dataloader_proj.py` – Dataset/Dataloader für Projektionen + Labels
- `Network_Architectures/` – Backbones (ResNet/VGG/EfficientNet)
- `config/` – YAML‑Konfigurationen + Sweep‑Definition

## Setup

### 1) Environment
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Secrets / Konfiguration
Lege eine `.env` Datei anhand von `.env.example` an:
```bash
cp .env.example .env
```

Für W&B:
- Setze `WANDB_API_KEY` (und optional `WANDB_PROJECT`, `WANDB_ENTITY`) in `.env`
- **Keine** API Keys committen.

## Datenformat (erwartet)
Das Dataset sucht in `dataloader_params.inputFolder` nach Unterordnern, die `d_combined.npy` enthalten und lädt zusätzlich alle `*.raw` Dateien in diesen Unterordnern.

Beispiel:
```
<DATA_ROOT>/
  0_gear1/
    0001.raw
    0002.raw
    ...
    d_combined.npy
  1_gear1/
    ...
```

> Details siehe `dataloader_proj.py` (`process_subdirectories()`, `loadProjections()`, `load_detectabilites()`).

## Training
Standard‑Run (Beispiel‑Config):
```bash
python trainer.py
```

Welche Config genutzt wird, ist im `trainer.py` über `@hydra.main(...)` definiert.
Du kannst eine andere Config auswählen, indem du dort `config_name="..."` setzt oder (empfohlen) das Script so anpasst, dass `--config-name` unterstützt wird.

## Sweeps (W&B)
```bash
wandb sweep config/sweep.yaml
wandb agent <SWEEP_ID>
```

## Reproduzierbarkeit (Empfehlungen)
- Objekt‑basierte Splits (keine Projektionen desselben Objekts in Train und Val/Test mischen)
- Pfade immer über Config/ENV, keine absoluten OS‑spezifischen Pfade committen
- Runs: Config‑Snapshot + Git Commit Hash loggen (W&B kann das automatisch)

## Lizenz
Dieses Repo ist unter der **Apache License 2.0** lizenziert (siehe `LICENSE`).

---
### Zitation / Kontext
Wenn du diese Arbeit verwendest, bitte entsprechend in deiner Arbeit/Publikation zitieren.
