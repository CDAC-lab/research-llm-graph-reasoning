import os
import json

class PromptUtils:
    @staticmethod
    def ensure_dir(d):
        if not os.path.exists(d):
            os.makedirs(d)

    @staticmethod
    def get_latest_prompt(prompt_dir):
        files = [f for f in os.listdir(prompt_dir) if f.startswith("v_") and f.endswith(".txt")]
        if not files:
            return None, 0
        versions = [int(f.split("v_")[1].split(".")[0]) for f in files]
        max_version = max(versions)
        with open(os.path.join(prompt_dir, f"v_{max_version}.txt"), "r", encoding="utf-8") as f:
            content = f.read()
            if "---human---" not in content:
                file_path = os.path.join(prompt_dir, f"v_{max_version}.txt")
                raise ValueError(f"Prompt file {file_path} missing '---human---' delimiter")
            system_part, human_part = content.split("---human---", 1)
            system_part = system_part.replace("---system---", "").strip()
            human_part = human_part.strip()
            return [("system", system_part), ("human", human_part)], max_version

    @staticmethod
    def save_new_prompt(prompt_text, version, prompt_dir):
        print(f"Saving new prompt version {version} to {prompt_dir}")
        print(f"Prompt text: {prompt_text}")
        prompt_text = "---system---\n" + prompt_text[0][1] + "\n---human---\n" + prompt_text[1][1]
        with open(os.path.join(prompt_dir, f"v_{version}.txt"), "w", encoding="utf-8") as f:
            f.write(prompt_text)

    @staticmethod
    def save_accuracy(version, accuracy, accuracy_dir):
        PromptUtils.ensure_dir(accuracy_dir)
        acc_file = os.path.join(accuracy_dir, f"accuracy_v{version}.json")
        with open(acc_file, "w", encoding="utf-8") as f:
            json.dump({"version": version, "accuracy": accuracy}, f, indent=2)

    @staticmethod
    def extract_refined_prompt(llm_response):
        marker = "### REFINED_PROMPT_START"
        if marker not in llm_response:
            return None
        refined = llm_response.split(marker, 1)[1]
        return refined.strip()
