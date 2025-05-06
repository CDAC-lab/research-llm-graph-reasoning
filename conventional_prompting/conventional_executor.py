from dotenv import load_dotenv
import os
import json
from datasets import load_dataset
from utils.dataset_utils import DatasetUtils
from utils.knowledge_graph_utils import KnowledgeGraphUtils
from utils.general_utils import GeneralUtils
from langchain_core.globals import set_debug, set_verbose, set_llm_cache
from langchain_community.cache import InMemoryCache
from langchain_core.prompts import ChatPromptTemplate
from langchain.models import get_llm
from langchain.structured_outputs import ConventionalResponse


class ConventionalExecutor:
    def __init__(self, dataset_name, is_debug=False, sample_question_indexes=None):
        self.dataset_name = dataset_name
        self.is_debug = is_debug
        self.dataset_config = None
        self.general_config = None

        if sample_question_indexes is None:
            sample_question_indexes = []
        self.sample_question_indexes = sample_question_indexes

        # Read .env file
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

    def load_dataset_configs(self):
        # Load the dataset related configs
        try:
            with open(f'configs/{self.dataset_name}.json') as f:
                self.dataset_config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file for {self.dataset_name} not found.")

    def load_general_configs(self):
        # Load the general configs
        try:
            with open('configs/general.json') as f:
                self.general_config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("General configuration file not found.")

    @staticmethod
    def get_conventional_execution_prompt(dataset_name, relationships_list, entity_classes_list):
        prompt = None
        if dataset_name == 'clutrr':
            prompt = f"""
            input story:
            {{input_story}}
            
            question:
            {{question}}
            
            First, create a knowledge graph by extracting facts from each sentence in the given input story. 
            Once this is done, I will pose a question. This question can be transformed into a triple (s, ?, o), where your primary task 
            is to determine the missing relation (’?’) that links the subject entity (’s’) to the object entity (’o’). To begin, focus on 
            the subject entity in this triple and choose the most relevant facts to expand from it. Step by step, progress towards the 
            object entity, ensuring that each selected fact contributes to creating a link between the subject and object entities. 
            Finally, utilize the established connection between the subject and object entities to answer the question.
            """
        return prompt

    def build_conventional_execution_chain(self):
        prompt = ChatPromptTemplate.from_template(
            self.get_conventional_execution_prompt(
                self.dataset_name,
                self.dataset_config['relationships_list'],
                self.dataset_config['entity_classes_list']
            )
        )
        llm_model = get_llm(
            llm_type=self.general_config["llm_type"],
            llm_model=self.general_config["llm_model"],
            api_key=self.openai_api_key
        )
        chain = prompt | llm_model.with_structured_output(schema=ConventionalResponse)
        return chain

    def load_questions_list(self, is_debug, sample_question_indexes):
        if self.dataset_name == 'clutrr':
            ds = load_dataset(
                path=self.dataset_config['dataset_path_in_hugging_face'],
                name=self.dataset_config['dataset_name_in_hugging_face']
            )
            list_dict_test, list_questions = DatasetUtils.pre_process_clutrr_dataset(
                dataset=ds, is_debug=is_debug, sample_question_indexes=sample_question_indexes)
            print(f"sample list_dict_test :- {list_dict_test[:10]}")
            print(f"sample list_questions :- {list_questions[:10]}")
            return list_dict_test, list_questions

    @staticmethod
    def chunk_list(list_questions, list_dict_test, chunk_size):
        for i in range(0, len(list_questions), chunk_size):
            yield list_questions[i:i + chunk_size], list_dict_test[i:i + chunk_size]

    def execute(self):
        # Load dataset configs
        self.load_dataset_configs()
        self.load_general_configs()

        # Load questions list
        list_dict_test, list_questions = self.load_questions_list(
            is_debug=self.is_debug, sample_question_indexes=self.sample_question_indexes)
        print(f"len of list_questions :- {len(list_questions)}")
        list_questions = list_questions[:200]  # Limit to 10 questions for testing
        list_dict_test = list_dict_test[:200]  # Limit to 10 questions for testing

        # build the chain
        set_debug(False)
        set_verbose(False)
        set_llm_cache(InMemoryCache())

        conventional_execution_chain = self.build_conventional_execution_chain()

        knowledge_graph_utils = KnowledgeGraphUtils(self.dataset_name)
        general_utils = GeneralUtils(self.dataset_name)

        batch_num = 0
        for questions_chunk, dicts_chunk in self.chunk_list(list_questions, list_dict_test,
                                                            self.dataset_config['batch_size']):
            # Execute the chain if final answer csv file does not exist
            if os.path.exists(
                    f"./outputs/conventional_answers/{self.dataset_name}_conventional_answers_b{batch_num}.csv"):
                print(f"⚠️ Final answer file already exists for batch {batch_num}. Skipping...")
                batch_num += 1
                continue
            else:
                print(f"started processing batch {batch_num} with {len(questions_chunk)} questions ... ")

                conventional_answers_questions_list = DatasetUtils.conventional_questions_for_prompt(
                    questions_chunk, dicts_chunk)

                conventional_answers_list = conventional_execution_chain.batch(
                    conventional_answers_questions_list,
                    config={
                        "max_concurrency": self.dataset_config["max_concurrency"]
                    }
                )

                general_utils.save_conventional_answers_as_csv(
                    conventional_answers_list,
                    dicts_chunk,
                    batch_num
                )

                print(f"finished processing batch {batch_num} successfully !")
                batch_num += 1

                # Cleanup
                del conventional_answers_list
                del conventional_answers_questions_list