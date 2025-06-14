# Graph Reasoning with LLMs

This repository demonstrates an experiment using large language models (LLMs) to extract information from short
narratives, convert that information into a knowledge graph, and use the graph to answer queries. The current
implementation focuses on the [CLUTRR dataset](https://huggingface.co/datasets/CLUTRR/v1), which tests reasoning about
family relationships.

## Overview

- **Purpose**: showcase how an LLM can build a knowledge graph from text and answer questions about the relationships it
  contains.
- **Entry point**: `main.py` chooses between a dynamic graph builder and a conventional prompting baseline. Set
  `is_conventional` to switch between the two and use `is_debug` to restrict the run to a few questions for quick
  testing.
- **Outputs**: generated graphs, intermediate CSV files, and final answers are written to the `outputs/` directory (
  which is listed in `.gitignore`).

## Environment setup

1. Create a Python virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies from the provided requirements file:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API key so the builders can load it:
   ```
   OPENAI_API_KEY=your-key-here
   ```

## Configuration

The `configs/` folder holds JSON configuration files.

- `configs/general.json` – defines the LLM type, model name, and number of workers.

  ```json
  {
    "llm_type": "openai",
    "llm_model": "gpt-4.1",
    "max_workers": 3
  }
  ```
- `configs/clutrr.json` – dataset hugging-face configs, batch size, allowed relationships, and entity classes.

  ```json
  {
    "dataset_path_in_hugging_face": "CLUTRR/v1",
    "dataset_name_in_hugging_face": "gen_train234_test2to10",
    "batch_size": 50,
    "max_concurrency": 5,
    "relationships_list": ["has_aunt", ...],
    "entity_classes_list": ["Aunt", ...]
  }
  ```

Adjust these files if you want to experiment with other models or datasets.

## Running the experiment

Run the project with:

```bash
python main.py
```

`main.py` currently sets `is_debug=True` and `is_conventional=False`. Edit those flags if you want to run the baseline
approach or process the entire dataset.

### Quick start

```bash
git clone <repo-url>
cd <repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # or install packages manually
echo "OPENAI_API_KEY=sk-..." > .env
python main.py
```

## Project workflow

1. Read the dataset and parse the stories.
2. Use the LLM to extract triples representing relationships.
3. Convert the triples into OWL files and perform queries.
4. Generate answers from the graph (and optionally revise them).

## Dynamic graph builder

The `dynamic_graph_building/dynamic_graph_builder.py` file orchestrates this
pipeline. It loads dataset-specific settings from `configs/` and performs the
following steps:

1. **Extraction prompt** – A detailed system prompt enumerates rules such as
   *Spouse-detection*, *Mother's-husband*, *Son-naming*, and others. These rules
   guide the LLM to output relationship triples in a fixed format. The allowed
   relationship types are loaded from `configs/clutrr.json`.
2. **Knowledge graph creation** – Extracted triples are saved as OWL files using
   `KnowledgeGraphUtils.save_llm_response_as_owl`.
3. **Path querying** – The graphs are queried to find chains connecting the
   subject and object in each question.
4. **Answer generation** – Another prompt turns these chains into final answers
   via `build_pre_revised_answers_generator_chain`. Each batch of results is
   written to `outputs/pre_revised_answers/`.

The builder processes the dataset in batches (`batch_size` in the config) and
skips any batch whose output already exists, allowing interrupted runs to be
resumed.

## Resources

- [CLUTRR Dataset](https://huggingface.co/datasets/CLUTRR/v1)
- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API](https://platform.openai.com/docs/)