# Development

## Install

Use Python 3.10 or newer. A virtual environment keeps the project dependencies isolated:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

The project depends on MuJoCo, NumPy, and Pydantic. The `dev` extra adds pytest and Ruff.

## Test and lint

```bash
python -m pytest -q
ruff check src tests
```

The tests use the bundled Panda model and mock inference; they do not contact a model service. `tests/test_llm_real.py` is an opt-in smoke test. It skips by default and requires the selected Ollama model to be installed and the local service to be running:

```bash
RECOVERA_RUN_REAL_LLM=1 \
RECOVERA_LLM_MODEL=llama3.2:3b \
RECOVERA_LLM_TIMEOUT=60 \
python -m pytest -s tests/test_llm_real.py -q
```

## Run from source

The editable install exposes the `recovera` command. The equivalent module invocation is:

```bash
python -m recovera.cli
python -m recovera.cli --scenario obstacle-replan --llm-provider mock
```

On macOS, launch the native MuJoCo passive viewer through the `mjpython` executable installed with MuJoCo:

```bash
mjpython -m recovera.cli --viewer --scenario obstacle-replan --llm-provider mock
```

## Saved output and generated files

Trial records are written to `results/` by default and are ignored by Git to keep machine-specific runs out of the source history. Pass `--results-dir PATH` to choose another location. MuJoCo generates the combined scene XML and compiled model cache beside the Panda MJCF; those generated files are ignored as well.

## Repository contents

- `src/recovera/`: application, environment, task, monitoring, planner, recovery, provider, and CLI.
- `assets/franka_emika_panda/`: Panda MJCF and upstream meshes, with upstream documentation and license.
- `assets/obstacles/table_lamp/`: lamp source model, scene meshes, and source/license note.
- `tests/`: headless unit and integration tests; the real Ollama smoke test is opt-in.
- `docs/images/`: a rendered MuJoCo scene used by the README.
