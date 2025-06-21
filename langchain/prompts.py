from pathlib import Path
from typing import List, Tuple


class Prompts:
    """Utility class to load prompt templates from text files."""

    BASE_DIR = Path(__file__).resolve().parent / "prompt_library"

    @staticmethod
    def _latest_version_file(dataset: str, task: str) -> Path:
        task_dir = Prompts.BASE_DIR / dataset / task
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
    def _load_string_prompt(dataset: str, task: str) -> str:
        path = Prompts._latest_version_file(dataset, task)
        return path.read_text()

    @staticmethod
    def _load_messages_prompt(dataset: str, task: str) -> List[Tuple[str, str]]:
        path = Prompts._latest_version_file(dataset, task)
        content = path.read_text()
        if "---human---" not in content:
            raise ValueError(f"Prompt file {path} missing '---human---' delimiter")
        system_part, human_part = content.split("---human---", 1)
        system_part = system_part.replace("---system---", "").strip()
        human_part = human_part.strip()
        return [("system", system_part), ("human", human_part)]

    @staticmethod
    def get_graph_prompt(dataset_name: str, relationships_list) -> str:
        template = Prompts._load_string_prompt(dataset_name, "graph")
        return template.format(relationships_list=relationships_list)

    @staticmethod
    def get_relationship_prompt(
        dataset_name: str, entity_classes_list
    ) -> List[Tuple[str, str]]:
        messages = Prompts._load_messages_prompt(dataset_name, "final_answer")
        system_msg = messages[0][1].format(entity_classes_list=entity_classes_list)
        return [("system", system_msg), messages[1]]

    @staticmethod
    def get_revision_prompt(dataset_name: str, entity_classes_list) -> str:
        template = Prompts._load_string_prompt(dataset_name, "revise")
        return template.format(entity_classes_list=entity_classes_list)

    @staticmethod
    def get_extraction_prompt(relationships_list) -> List[Tuple[str, str]]:
        # extraction prompts are only dataset specific to 'clutrr'
        dataset_name = "clutrr"
        messages = Prompts._load_messages_prompt(dataset_name, "triple_extraction")
        system_msg = messages[0][1].format(relationships_list=relationships_list)
        return [("system", system_msg), messages[1]]
