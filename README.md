# Auditable ReAct-Pandas

Minimal runnable implementation of Auditable ReAct-Pandas for multi-table QA.
The repository contains only the current solver, its model client, evaluation
normalizer, focused unit tests, and a placeholder for the separately hosted
clean data archive.

## Repository Layout

- `auditable_react_pandas/`: solver package.
- `run.py`: command-line entry point.
- `tests/`: model-free unit tests.
- `data/clean_1818_392/`: metadata and expected location for downloaded data.

## Data

Large JSON data files are not committed to GitHub. Download the clean trusted
runnable data archive from Google Drive:

```text
https://drive.google.com/file/d/1JN7cUt-FdunkC5iIUU0K1UM1CVY95XdB/view?usp=sharing
```

Place `auditable_react_pandas_clean_data.tar.gz` in the repository root and
extract it:

```bash
tar -xzf auditable_react_pandas_clean_data.tar.gz
```

This creates:

- `data/clean_1818_392/two_clean_1818.json.gz`
- `data/clean_1818_392/multi_clean_392.json.gz`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the default DeepSeek run, set:

```bash
export DEEPSEEK_API_KEY=<your-key>
```

OpenAI-compatible models can be used with `OPENAI_API_KEY` and optionally
`OPENAI_BASE_URL`.

## Run

The runner can read `.json` and `.json.gz` datasets directly:

```bash
python run.py \
  --dataset_path data/clean_1818_392/two_clean_1818.json.gz \
  --log_dir local_outputs/smoke_two \
  --model_name deepseek-v4-flash \
  --stop_at 5 \
  --n_worker 1
```

Outputs are written under `local_outputs/`, which is ignored by Git.

## Tests

```bash
python -m unittest discover -s tests -p 'test_*.py'
```
