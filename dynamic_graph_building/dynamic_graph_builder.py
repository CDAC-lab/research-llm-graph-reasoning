from dotenv import load_dotenv
import os
import json
from datasets import load_dataset
from utils.dataset_utils import DatasetUtils
from utils.knowledge_graph_utils import KnowledgeGraphUtils
from langchain.structured_outputs import KnowledgeGraph
from langchain.models import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.globals import set_debug, set_verbose, set_llm_cache
from langchain_community.cache import InMemoryCache
import pandas as pd


class DynamicGraphBuilder:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.dataset_config = None
        self.general_config = None

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

    def load_questions_list(self):
        if self.dataset_name == 'clutrr':
            ds = load_dataset(
                path=self.dataset_config['dataset_path_in_hugging_face'],
                name=self.dataset_config['dataset_name_in_hugging_face']
            )
            list_dict_test, list_questions = DatasetUtils.pre_process_clutrr_dataset(ds)
            print(f"sample list_dict_test :- {list_dict_test[:10]}")
            print(f"sample list_questions :- {list_questions[:10]}")
            return list_dict_test, list_questions

    @staticmethod
    def get_graph_prompt(dataset_name, relationships_list, entity_classes_list):
        """Generates a prompt for the LLM to create a knowledge graph."""
        prompt = None
        if dataset_name == 'clutrr':
            prompt = """
                You are an expert in knowledge graph construction regarding families and relationships.
                You have to create a knowledge graph extracting all facts from the following statement,
                \n\n{statement}\n\n
                The relationships in the graph has to be in this list : \n\n """ + str(relationships_list) + \
                     """\n\n The classes of the entities in the graph has to be in this list: \n\n""" + str(
                entity_classes_list) + \
                     """\n\n Infer the inverse relationships and add them to the graph as well. \n\n"""

        return prompt

    def build_chain(self):
        prompt = ChatPromptTemplate.from_template(
            self.get_graph_prompt(
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
        chain = prompt | llm_model.with_structured_output(schema=KnowledgeGraph)
        return chain

    def execute(self):
        # Load dataset configs
        self.load_dataset_configs()
        self.load_general_configs()

        # Load questions list
        list_dict_test, list_questions = self.load_questions_list()
        list_questions = list_questions[:10]  # Limit to 10 questions for testing

        # build the chain
        set_debug(False)
        set_verbose(False)
        set_llm_cache(InMemoryCache())
        chain = self.build_chain()

        # Execute the chain
        graphs_list = chain.batch(
            list_questions,
            config={
                "max_concurrency": self.dataset_config["max_concurrency"]
            }
        )
        del list_questions

        # Save Knowledge Graphs
        knowledge_graph_utils = KnowledgeGraphUtils(self.dataset_name)
        knowledge_graph_utils.save_llm_response_as_owl(graphs_list)
