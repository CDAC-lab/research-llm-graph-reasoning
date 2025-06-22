import os
import re
from dotenv import load_dotenv
import json
import pandas as pd
from datasets import load_dataset
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.globals import set_debug, set_verbose, set_llm_cache
from langchain_community.cache import InMemoryCache
import xml.etree.ElementTree as ET

from langchain.models import get_llm
from utils.prompt_utils import PromptUtils
from utils.llm_utils import LlmUtils
from utils.knowledge_graph_utils import KnowledgeGraphUtils


def extract_triples_from_kg_obj(kg):
    """Extract list of (entity_1, edge, entity_2) triples from a KnowledgeGraph object"""
    return [(entry.entity_1.strip().lower(), entry.edge.strip().lower().replace('_', ' '), entry.entity_2.strip().lower()) for entry in kg.graph]

def calculate_accuracy(predictions, ground_truths):
    """
    predictions: list of KnowledgeGraph objects
    ground_truths: list of KnowledgeGraph objects (same length/order)

    Returns: accuracy (float)
    """
    correct = 0
    total = len(ground_truths)
    for pred_kg, gt_kg in zip(predictions, ground_truths):
        pred_triples = set(extract_triples_from_kg_obj(pred_kg))
        gt_kg = [(item[0].strip().lower(), item[1].strip().lower(), item[2].strip().lower()) for item in gt_kg]
        print(f"Predicted triples: {pred_triples}")
        print(f"Ground truth triples: {gt_kg}")
        gt_triples = set(gt_kg)
        if gt_triples.issubset(pred_triples):
            correct += 1
    return correct / total if total > 0 else 0.0


def load_messages_from_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    system_msg = root.find('system').text.strip()
    human_msg = root.find('human').text.strip()
    return system_msg, human_msg


def build_correction_messages(system_msg, human_msg, statement, expected, received, cur_prompt):
    filled_human_msg = human_msg.format(
        statement=statement,
        expected=expected,
        received=received,
        cur_prompt=cur_prompt
    )
    return [
        SystemMessage(content=system_msg),
        HumanMessage(content=filled_human_msg)
    ]


def get_clutrr_test_df(sample_question_indexes):
    ds = load_dataset(
        path="CLUTRR/v1",
        name="gen_train234_test2to10"
    )
    df_test = ds['test'].to_pandas()
    df_test = df_test.iloc[sample_question_indexes]
    return df_test


def get_chain(llm_model, prompt):
    chain = prompt | llm_model | StrOutputParser()
    return chain


def convert_string_to_triples(triples_string):
    # Use regex to find all patterns like (X, Y, Z)
    pattern = r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)'
    matches = re.findall(pattern, triples_string)

    # Convert matches to list of tuples, stripping whitespace
    result = [(s.strip(), p.strip(), o.strip()) for s, p, o in matches]

    return result

def _sanitize_model_name(llm_model: str) -> str:
    """Sanitize model name to be a valid folder name."""
    name = llm_model.replace("-", "_").replace(".", "_").replace(" ", "_")
    name = re.sub(r'[\\/:*?"<>|]', '_', name)
    return name


def main():
    dataset = "clutrr"
    task = "triple_extraction"
    sample_question_indexes = [
        0, 2, 5, 7, 1, 12, 39, 42, 40, 41, 140, 38, 46, 146, 147, 148, 149, 166, 184, 145, 150, 223, 224, 226, 227, 242,
        347, 222, 225, 502, 510, 500, 503, 505, 450, 501, 507, 514, 524, 513, 518, 512, 523, 517, 526, 671, 674, 668,
        669, 670, 711, 667, 673, 804, 805, 803, 807, 827, 852, 806, 809, 935, 938, 928, 932, 927, 942, 926, 931
    ]
    relationships_list = [
        "has_aunt",
        "has_uncle",
        "has_brother",
        "has_sister",
        "has_daughter",
        "has_son",
        "has_daughter_in_law",
        "has_son_in_law",
        "has_father",
        "has_mother",
        "has_father_in_law",
        "has_mother_in_law",
        "has_granddaughter",
        "has_grandson",
        "has_grandfather",
        "has_grandmother",
        "has_husband",
        "has_wife",
        "has_nephew",
        "has_niece"
    ]

    # Load config
    with open("../configs/prompt_enhancement.json", "r", encoding="utf-8") as f:
        CONFIG = json.load(f)

    PROMPT_DIR = CONFIG["PROMPT_DIR"]
    SEED_PROMPT_FILE = CONFIG["SEED_PROMPT_FILE"]
    GROUND_TRUTH_FILE = CONFIG["GROUND_TRUTH_FILE"]
    MAX_LOOPS = CONFIG["MAX_LOOPS"]
    SAMPLE_SIZE = CONFIG["SAMPLE_SIZE"]
    ACCURACY_DIR = CONFIG["ACCURACY_DIR"]
    CORRECTION_TEMPLATE_FILE = CONFIG["CORRECTION_TEMPLATE_FILE"]
    OPENAI_MODEL = CONFIG["OPENAI_MODEL"]

    PROMPT_DIR = f"{PROMPT_DIR}/{dataset}/{_sanitize_model_name(OPENAI_MODEL)}/{task}"
    SEED_PROMPT_FILE = SEED_PROMPT_FILE.replace("{dataset}", dataset).replace("{task}", task)
    GROUND_TRUTH_FILE = GROUND_TRUTH_FILE.replace("{dataset}", dataset).replace("{task}", task)
    ACCURACY_DIR = ACCURACY_DIR.replace("{dataset}", dataset).replace("{task}", task)
    CORRECTION_TEMPLATE_FILE = CORRECTION_TEMPLATE_FILE.replace("{dataset}", dataset).replace("{task}", task)

    PromptUtils.ensure_dir(PROMPT_DIR)
    print(PROMPT_DIR)
    PromptUtils.ensure_dir(ACCURACY_DIR)
    system_msg, human_msg = load_messages_from_xml(CORRECTION_TEMPLATE_FILE)

    # Seed or get latest prompt
    files = [f for f in os.listdir(PROMPT_DIR) if f.startswith("v_") and f.endswith(".txt")]
    if not files:
        with open(SEED_PROMPT_FILE, "r", encoding="utf-8") as f:
            seed_prompt = f.read()
        version = 1
        latest_prompt = seed_prompt
        PromptUtils.save_new_prompt(seed_prompt, version, PROMPT_DIR)
        print("Initialized v_1.txt from seed_prompt.txt.")
    else:
        latest_prompt, version = PromptUtils.get_latest_prompt(PROMPT_DIR)
        print(f"Loaded v_{version}.txt from {PROMPT_DIR}.")

    if task == "triple_extraction":
        system_msg = latest_prompt[0][1].format(relationships_list=relationships_list)
        latest_prompt = [("system", system_msg), latest_prompt[1]]
    else:
        return

        # Load ground truth CSV
    ground_truth_df = pd.read_csv(GROUND_TRUTH_FILE)

    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # build the chain
    set_debug(False)
    set_verbose(False)
    set_llm_cache(InMemoryCache())
    llm = get_llm(
        llm_type='openai',
        llm_model=OPENAI_MODEL,
        api_key=openai_api_key
    )
    chain = get_chain(llm, ChatPromptTemplate(latest_prompt))

    # --- Load CLUTRR dataset ---
    print("Loading CLUTRR test samples from Huggingface...")
    df = get_clutrr_test_df(sample_question_indexes)
    print(f"Loaded {len(df)} CLUTRR test samples.")

    # Initial accuracy eval and save
    sample = df.copy()
    batch_input = [{"statement": row['story']} for _, row in sample.iterrows()]
    print("Evaluating initial accuracy...")
    predictions = chain.batch(
        batch_input,
        config={
            "max_concurrency": 3
        }
    )
    print("Initial predictions generated.")
    predictions = [KnowledgeGraphUtils.extract_triples_from_llm_str_output(p) for p in predictions]
    print(f"Predictions converted to triples format. eg :- {predictions[:3]}")
    ground_truth_df.loc[:, 'ground_truth'] = ground_truth_df['ground_truth'].apply(convert_string_to_triples)

    prev_accuracy = calculate_accuracy(predictions, ground_truth_df["ground_truth"].tolist())
    PromptUtils.save_accuracy(version, prev_accuracy, ACCURACY_DIR)
    print(f"Initial accuracy for version {version}: {prev_accuracy:.2%}")

    # for loop_count in range(MAX_LOOPS):
    #     sample = df.sample(n=SAMPLE_SIZE, random_state=loop_count + 1)
    #     batch_input = [(latest_prompt, row['statement']) for _, row in sample.iterrows()]
    #     predictions = LlmUtils.batch_llm_generate(llm, batch_input)
    #     current_accuracy = calculate_accuracy(predictions, sample["expected_answer"].tolist())
    #     print(f"Loop {loop_count + 1}: Version {version}, Accuracy {current_accuracy:.2%}")
    #
    #     if current_accuracy == 1.0:
    #         print("All results accurate. Finished!")
    #         break
    #
    #     failed_idx = None
    #     for idx, (p, g) in enumerate(zip(predictions, sample["expected_answer"])):
    #         if p.strip().lower() != g.strip().lower():
    #             failed_idx = idx
    #             break
    #
    #     if failed_idx is None:
    #         print("No failed cases found, stopping loop.")
    #         break
    #
    #     # Prepare correction messages using XML template
    #     correction_messages = build_correction_messages(
    #         system_msg, human_msg,
    #         sample.iloc[failed_idx]["statement"],
    #         sample.iloc[failed_idx]["expected_answer"],
    #         predictions[failed_idx],
    #         latest_prompt
    #     )
    #
    #     # Single LLM call for correction (not batch)
    #     correction_pred = llm.invoke(correction_messages).content
    #     refined_prompt = PromptUtils.extract_refined_prompt(correction_pred)
    #     if not refined_prompt:
    #         print("LLM did not return a refined prompt. Stopping.")
    #         break
    #
    #     # Evaluate refined prompt accuracy (batch)
    #     test_batch_input = [(refined_prompt, row['statement']) for _, row in sample.iterrows()]
    #     refined_predictions = LlmUtils.batch_llm_generate(llm, test_batch_input)
    #     refined_accuracy = calculate_accuracy(refined_predictions, sample["expected_answer"].tolist())
    #     print(f"Refined prompt test accuracy: {refined_accuracy:.2%}")
    #
    #     if refined_accuracy > current_accuracy:
    #         version += 1
    #         PromptUtils.save_new_prompt(refined_prompt, version, PROMPT_DIR)
    #         PromptUtils.save_accuracy(version, refined_accuracy, ACCURACY_DIR)
    #         latest_prompt = refined_prompt
    #         prev_accuracy = refined_accuracy
    #         print(f"Prompt improved! Saved as prompt_v{version}.txt with accuracy {refined_accuracy:.2%}.")
    #     else:
    #         print("Refined prompt did not improve accuracy. Keeping previous prompt.")


if __name__ == "__main__":
    main()
