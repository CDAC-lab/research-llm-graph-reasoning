from dotenv import load_dotenv
import os
import json
from datasets import load_dataset
from utils.dataset_utils import DatasetUtils
from utils.knowledge_graph_utils import KnowledgeGraphUtils
from utils.general_utils import GeneralUtils
from langchain.structured_outputs import KnowledgeGraph, RelationshipResponse, RevisedResponse
from langchain.models import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import Prompts
from langchain_core.output_parsers import StrOutputParser
from langchain_core.globals import set_debug, set_verbose, set_llm_cache
from langchain_community.cache import InMemoryCache
import logging
import pandas as pd


class DynamicGraphBuilder:
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


    def build_knowledge_graph_str_extractor_chain(self, llm_model):
        # prompt = ChatPromptTemplate.from_template(
        #     Prompts.get_graph_prompt(
        #         self.dataset_name,
        #         self.dataset_config["relationships_list"]
        #     )
        # )
        prompt = ChatPromptTemplate(
            Prompts.get_triple_extraction_prompt(
                self.general_config["llm_model"],
                self.dataset_config["relationships_list"]
            )
        )

        chain = prompt | llm_model | StrOutputParser()
        return chain

    def build_pre_revised_answers_generator_chain(self, llm_model):
        prompt = ChatPromptTemplate(
            Prompts.get_final_answer_prompt(
                self.dataset_name,
                self.general_config["llm_model"],
                self.dataset_config["entity_classes_list"],
            )
        )

        chain = prompt | llm_model | StrOutputParser()
        return chain

    def build_revised_answers_generator_chain(self, llm_model):
        prompt = ChatPromptTemplate.from_template(
            Prompts.get_revision_prompt(
                self.dataset_name,
                self.general_config["llm_model"],
                self.dataset_config["entity_classes_list"],
            )
        )

        chain = prompt | llm_model.with_structured_output(schema=RevisedResponse)
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
        list_dict_test, list_questions = self.load_questions_list(
            is_debug=self.is_debug, sample_question_indexes=self.sample_question_indexes)
        print(f"len of list_questions :- {len(list_questions)}")
        # list_questions = list_questions[:200]  # Limit questions for testing
        # list_dict_test = list_dict_test[:200]  # Limit questions for testing

        # build the chain
        set_debug(False)
        set_verbose(False)
        set_llm_cache(InMemoryCache())

        if self.general_config["llm_type"] != "ollama":
            llm_model = get_llm(
                llm_type=self.general_config["llm_type"],
                llm_model=self.general_config["llm_model"]
            )

        else:
            llm_model = get_llm(
                llm_type=self.general_config["llm_type"],
                llm_model=self.general_config["llm_model"],
                api_key=self.openai_api_key
            )

        knowledge_graph_str_extractor_chain = self.build_knowledge_graph_str_extractor_chain(llm_model)
        pre_revised_answers_generator_chain = self.build_pre_revised_answers_generator_chain(llm_model)
        # revised_answers_generator_chain = self.build_revised_answers_generator_chain(llm_model)

        knowledge_graph_utils = KnowledgeGraphUtils(self.dataset_name)
        general_utils = GeneralUtils(self.dataset_name)

        batch_num = 0
        for questions_chunk, dicts_chunk in self.chunk_list(list_questions, list_dict_test,
                                                            self.dataset_config['batch_size']):

            # Execute the chain if final answer csv file does not exist
            # if os.path.exists(f"./outputs/revised_answers/{self.dataset_name}_revised_answers_b{batch_num}.csv"):
            if os.path.exists(
                    f"./outputs/pre_revised_answers/{self.dataset_name}_pre_revised_answers_b{batch_num}.csv"):
                print(f"⚠️ Final answer file already exists for batch {batch_num}. Skipping...")
                batch_num += 1
                continue
            else:
                print(f"started processing batch {batch_num} with {len(questions_chunk)} questions ... ")

                graphs_list = []

                if self.general_config["llm_type"] == "ollama":
                    messages = [
                        ("system", ("You are an expert in relationship triple construction regarding families and "
                                    "relationships as (person_A, relationship, person_B). You have to create "
                                    "relationship triples extracting all facts from the user's statement. \n\n"
                                    f"The relationships in the graph has to be in this list : {str(self.dataset_config['relationships_list'])} \n\n"
                                    f"Infer the inverse relationships and add them to the response as well.")),
                        ("human", questions_chunk[0]["statement"]),
                    ]
                    graph_str = knowledge_graph_str_extractor_chain.invoke(
                        messages
                    )
                    graph_str = graph_str.content
                    print(f"Graph: {graph_str}")
                    graphs_list.append(KnowledgeGraphUtils.extract_triples_from_llm_str_output(graph_str))
                    print("-----------------------------------------------------")

                else:
                    graphs_str_list = knowledge_graph_str_extractor_chain.batch(
                        questions_chunk,
                        config={
                            "max_concurrency": self.dataset_config["max_concurrency"]
                        }
                    )

                    # for graph_str in graphs_str_list:
                    #     print(f"Graph: {graph_str}")
                    #     graphs_list.append(KnowledgeGraphUtils.extract_triples_from_llm_str_output(graph_str))
                    #     print("-----------------------------------------------------")

                # Save Knowledge Graphs
                knowledge_graph_utils.save_llm_response_as_owl(graphs_list, batch_num)
                del graphs_list
                del graphs_str_list

                # Query the knowledge graph
                output_file_path = knowledge_graph_utils.query_knowledge_graph_batch(
                    dicts_chunk,
                    batch_num,
                    self.general_config['max_workers']
                )

                # Filter the shortest paths
                output_file_path = general_utils.filter_shortest_paths(batch_num)

                # Prepare the prompt for the final answer generator
                chain_df = pd.read_csv(output_file_path)
                pre_revised_answers_questions_list = DatasetUtils.pre_process_pre_revised_questions_for_prompt(chain_df)
                pre_revised_answers_list = pre_revised_answers_generator_chain.batch(
                    pre_revised_answers_questions_list,
                    config={
                        "max_concurrency": self.dataset_config["max_concurrency"]
                    }
                )

                # Save the final answers
                pre_revised_answers_file = general_utils.save_pre_revised_answers_as_csv(
                    pre_revised_answers_list,
                    chain_df,
                    batch_num
                )
                del pre_revised_answers_questions_list
                del pre_revised_answers_list

                # # Generate revised answers
                # # pre_revised_answers_file = f"./outputs/pre_revised_answers/{self.dataset_name}_pre_revised_answers_b{batch_num}.csv"  # TODO: delete this. Used for testing
                # chain_df = pd.read_csv(pre_revised_answers_file)
                # revised_answers_questions_list = DatasetUtils.pre_process_revised_questions_for_prompt(chain_df)
                # revised_answers_list = revised_answers_generator_chain.batch(
                #     revised_answers_questions_list,
                #     config={
                #         "max_concurrency": self.dataset_config["max_concurrency"]
                #     }
                # )
                #
                # general_utils.save_revised_answers_as_csv(
                #     revised_answers_list,
                #     chain_df,
                #     batch_num
                # )

                print(f"finished processing batch {batch_num} successfully !")
                batch_num += 1

                # Cleanup
                del chain_df
                # del revised_answers_questions_list
                # del revised_answers_list

        del list_questions
        del list_dict_test
