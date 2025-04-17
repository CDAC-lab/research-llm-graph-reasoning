from dotenv import load_dotenv
import os
import json
from datasets import load_dataset
from utils.dataset_utils import DatasetUtils
from utils.knowledge_graph_utils import KnowledgeGraphUtils
from utils.general_utils import GeneralUtils
from langchain.structured_outputs import KnowledgeGraph, RelationshipResponse
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

    @staticmethod
    def get_relationship_prompt(dataset_name, entity_classes_list):
        """Generates a prompt for the LLM to create a knowledge graph."""
        prompt = None
        if dataset_name == 'clutrr':
            prompt = f"""
            You are an expert in understanding family and social relationships.
            Given a Input sequence of relationships between people, determine the relationship between the first person and the last person in the chain.
            Choose the best answer from the following list and respond with only one word that best completes the sentence:
            
            Valid answers:
            {entity_classes_list}
            
            Format your response exactly as:
            "{{last}} is {{first}}'s ______."
            (Fill in the blank with only one word from the list above.)
            
            Input sequence:
            {{relation_chain}}
            
            Output:
            """

        return prompt

    def build_knowledge_graph_extractor_chain(self):
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

    def build_final_answer_generator_chain(self):
        prompt = ChatPromptTemplate.from_template(
            self.get_relationship_prompt(
                self.dataset_name,
                self.dataset_config['entity_classes_list']
            )
        )
        llm_model = get_llm(
            llm_type=self.general_config["llm_type"],
            llm_model=self.general_config["llm_model"],
            api_key=self.openai_api_key
        )
        chain = prompt | llm_model.with_structured_output(schema=RelationshipResponse)
        return chain

    @staticmethod
    def chunk_list(list_questions, list_dict_test, chunk_size):
        for i in range(0, len(list_questions), chunk_size):
            yield list_questions[i:i + chunk_size], list_dict_test[i:i + chunk_size]

    def execute(self):
        # Load dataset configs
        self.load_dataset_configs()
        self.load_general_configs()

        # Load questions list
        list_dict_test, list_questions = self.load_questions_list()
        # list_questions = list_questions[:200]  # Limit to 10 questions for testing
        # list_dict_test = list_dict_test[:200]  # Limit to 10 questions for testing

        # build the chain
        set_debug(False)
        set_verbose(False)
        set_llm_cache(InMemoryCache())
        knowledge_graph_extractor_chain = self.build_knowledge_graph_extractor_chain()
        final_answer_generator_chain = self.build_final_answer_generator_chain()

        knowledge_graph_utils = KnowledgeGraphUtils(self.dataset_name)
        general_utils = GeneralUtils(self.dataset_name)

        batch_num = 0
        for questions_chunk, dicts_chunk in self.chunk_list(list_questions, list_dict_test, self.dataset_config['batch_size']):
            print(f"started processing batch {batch_num} with {len(questions_chunk)} questions ... ")
            # Execute the chain
            graphs_list = knowledge_graph_extractor_chain.batch(
                questions_chunk,
                config={
                    "max_concurrency": self.dataset_config["max_concurrency"]
                }
            )

            # Save Knowledge Graphs
            knowledge_graph_utils.save_llm_response_as_owl(graphs_list, batch_num)
            del graphs_list

            # Query the knowledge graph
            output_file_path = knowledge_graph_utils.query_knowledge_graph_batch(
                dicts_chunk,
                batch_num,
                self.general_config['max_workers']
            )

            # Prepare the prompt for the final answer generator
            chain_df = pd.read_csv(output_file_path)
            final_answer_questions_list = DatasetUtils.pre_process_clutrr_chains_for_prompt(chain_df)
            final_answers_list = final_answer_generator_chain.batch(
                final_answer_questions_list,
                config={
                    "max_concurrency": self.dataset_config["max_concurrency"]
                }
            )

            # Save the final answers
            general_utils.save_final_answer_as_csv(
                final_answers_list,
                chain_df,
                batch_num
            )

            print(f"finished processing batch {batch_num} successfully !")
            batch_num += 1

            # Cleanup
            del chain_df
            del final_answer_questions_list
            del final_answers_list

        del list_questions
        del list_dict_test