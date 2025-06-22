from pathlib import Path
from typing import List, Tuple
import re


class Prompts:
    """Utility class to load prompt templates from text files."""

    BASE_DIR = Path(__file__).resolve().parent / "prompt_library"

    @staticmethod
    def _sanitize_model_name(llm_model: str) -> str:
        """Sanitize model name to be a valid folder name."""
        name = llm_model.replace("-", "_").replace(".", "_").replace(" ", "_")
        name = re.sub(r'[\\/:*?"<>|]', '_', name)
        return name

    @staticmethod
    def _latest_version_file(dataset: str, llm_model: str, task: str) -> Path:
        llm_dir = Prompts._sanitize_model_name(llm_model)
        task_dir = Prompts.BASE_DIR / dataset / llm_dir / task
        if not task_dir.exists():
            raise FileNotFoundError(f"Prompt directory {task_dir} does not exist")
        files = sorted(
            task_dir.glob("v_*.txt"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        if not files:
            raise FileNotFoundError(f"No prompt files found in {task_dir}")
        return files[-1]

    @staticmethod
    def _load_string_prompt(dataset: str, llm_model: str, task: str) -> str:
        path = Prompts._latest_version_file(dataset, llm_model, task)
        return path.read_text(encoding="utf-8")

    @staticmethod
    def _load_messages_prompt(dataset: str, llm_model: str, task: str) -> List[Tuple[str, str]]:
        path = Prompts._latest_version_file(dataset, llm_model, task)
        content = path.read_text(encoding="utf-8")
        if "---human---" not in content:
            raise ValueError(f"Prompt file {path} missing '---human---' delimiter")
        system_part, human_part = content.split("---human---", 1)
        system_part = system_part.replace("---system---", "").strip()
        human_part = human_part.strip()
        return [("system", system_part), ("human", human_part)]

    @staticmethod
    def get_graph_prompt(dataset_name: str, llm_model: str, relationships_list) -> str:
        template = Prompts._load_string_prompt(dataset_name, llm_model, "graph")
        return template.format(relationships_list=relationships_list)

    @staticmethod
    def get_final_answer_prompt(
        dataset_name: str, llm_model: str, entity_classes_list
    ) -> List[Tuple[str, str]]:
        messages = Prompts._load_messages_prompt(dataset_name, llm_model, "final_answer")
        system_msg = messages[0][1].format(entity_classes_list=entity_classes_list)
        human_msg = messages[1][1].format(entity_classes_list=entity_classes_list)
        return [("system", system_msg), ("human", human_msg)]

    @staticmethod
    def get_revision_prompt(dataset_name: str, llm_model: str, entity_classes_list) -> str:
        template = Prompts._load_string_prompt(dataset_name, llm_model, "revise")
        return template.format(entity_classes_list=entity_classes_list)

    @staticmethod
    def get_triple_extraction_prompt(llm_model: str, relationships_list) -> List[Tuple[str, str]]:
        # extraction prompts are only dataset specific to 'clutrr'
        dataset_name = "clutrr"
        messages = Prompts._load_messages_prompt(dataset_name, llm_model, "triple_extraction")
        system_msg = messages[0][1].format(relationships_list=relationships_list)
        return [("system", system_msg), messages[1]]
