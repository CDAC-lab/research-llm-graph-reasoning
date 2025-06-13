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

    @staticmethod
    def get_graph_prompt(dataset_name, relationships_list):
        """Generates a prompt for the LLM to create a knowledge graph."""
        prompt = None
        if dataset_name == 'clutrr':
            prompt = """
                You are an expert in relationship triple construction regarding families and relationships as (person_A, relationship, person_B).
                You have to create relationship triples extracting all facts from the following statement,
                \n\n{statement}\n\n
                The relationships in the graph has to be in this list : \n\n """ + str(relationships_list) + \
                     """\n\n Infer the inverse relationships and add them to the response as well. \n\n"""

        return prompt

    @staticmethod
    def get_relationship_prompt(dataset_name, entity_classes_list):
        """Generates a prompt for the LLM to create a knowledge graph."""
        prompt = None
        if dataset_name == 'clutrr':
            prompt = f"""
            You are an expert in understanding family and social relationships. You are given:

            1. A list of relationships between individuals in the form of (PersonA, Relationship, PersonB).
            2. A predefined list of valid relationship types. 
            
            Your task is to determine the relationship ({{first_person}}, ?, {{last_person}}), such that:
            **“{{last_person}} is the {{first_person}}'s ____.”**
            
            Follow these steps:
            
            2. Go through the predefined list of valid relationship types and grasp the meaning of each term. Consider this list as a case-insensitive list of relationship terms.
            3. Traverse the list of relationships step-by-step. At each step, explain how the relationship between {{first_person}} and the current person evolves.
            4. Use only terms from the valid list to describe each intermediate and final relationship. **Do not infer social roles or make assumptions that are not strictly derived from the relationship sequence.**
            5. Provide your final answer in one word, **strictly** from the valid list, and ensure it is logically consistent with your reasoning steps.
            6. **Do not reinterpret or relabel the final relationship. The final answer must match exactly the relationship logically derived in the final step.** Do not apply social or cultural heuristics like “sibling of a grandson is niece.” If your final reasoning says “granddaughter,” your answer must be exactly “granddaughter.”
            
            Valid answers:
            {entity_classes_list}
            
            Input sequence:
            {{relation_str}}
            
            Return your response as a JSON object in the following format:
            
            ```json keys :- 
              "reason": ""<step-by-step reasoning using only valid relationship terms>",
              "answer": "<single-word answer from valid list>"
            ```
            """

        return prompt

    @staticmethod
    def get_revision_prompt(dataset_name, entity_classes_list):
        prompt = None
        if dataset_name == 'clutrr':
            prompt = f"""
            You are an expert in understanding family relationships. You are given:

            1. A list of relationships between individuals in the form of (PersonA, Relationship, PersonB).
            2. A selected answer for the relationship ({{first_person}}, ?, {{last_person}}).
            3. Reasoning steps to derive the answer.

            Your task is to check the correctness of the provided answer and the previous reasoning steps and, if necessary, revise it. You will also provide a step-by-step reasoning for your revised answer.

            Follow these steps:

            1. Go through the predefined list of valid relationship types and grasp the meaning of each term. Consider this list as a case-insensitive list of relationship terms.
            2. Traverse the list of relationships and the previous reasoning steps step-by-step. At each step, try to identify logical errors in the defined relationships. Correct them if necessary in the revised answer and reasoning.
            3. When giving the revised answer and reasoning, use only terms from the valid list to describe each intermediate and final relationship. **Do not infer social roles or make assumptions that are not strictly derived from the relationship sequence.**
            4. Provide your final answer in one word, **strictly** from the valid list, and ensure it is logically consistent with your reasoning steps.

            Valid answers:
            {entity_classes_list}

            Input sequence:
            {{relation_str}}

            Selected answer:
            {{selected_answer}}

            Reasoning steps:
            {{reasoning_steps}}

            Return your response as a JSON object in the following format:

            ```json keys :-
               "previous_reasoning_errors": "<list of errors found in the previous reasoning steps>",
               "reason": ""<step-by-step reasoning for the revised answer>",
               "revised_answer": "<single-word answer from valid list>"
            ```
            """

        return prompt

    def build_knowledge_graph_str_extractor_chain(self):
        # prompt = ChatPromptTemplate.from_template(
        #     self.get_graph_prompt(
        #         self.dataset_name,
        #         self.dataset_config['relationships_list']
        #     )
        # )
        prompt = ChatPromptTemplate([
            ("system", f"""
        <system>
          <role>
            You are an expert in natural language understanding and human relationship extraction.
            Your purpose is to read a user-provided statement and output clean, name-only triples.
          </role>

          <behavior>
            <rule name="Spouse-detection">
              Whenever you see “[Name] fixed her husband [OtherName]” or “[Name] cooked his wife [OtherName] …”,
              infer a spousal relationship:
              (Name, has_husband, OtherName) and (OtherName, has_wife, Name) if “her husband”,
              or (Name, has_wife, OtherName) and (OtherName, has_husband, Name) if “his wife.”
            </rule>

            <rule name="Mother’s-husband">
              Whenever you see a phrase of the form “&lt;bracketed Name&gt;’s mother’s husband [OtherName]”,
              infer that OtherName is that bracketed Name’s father:
              (Name, has_father, OtherName) and (OtherName, has_son, Name).
            </rule>

            <rule name="Father’s">
              Whenever you see a phrase of the form “&lt;bracketed Name&gt;’s father, [OtherName]”,
              infer that OtherName is that bracketed Name’s father:
              (Name, has_father, OtherName) and (OtherName, has_daughter, Name).
            </rule>

            <rule name="Son-naming">
              Whenever you see “&lt;bracketed Name&gt;’s son, [OtherName]” (for example, “John’s son, [Bryan]”),
              produce (OtherName, has_father, Name) and (Name, has_son, OtherName).
            </rule>

            <rule name="Dad/Father-naming">
              Whenever you see “&lt;bracketed Name&gt;’s dad, [OtherName]” (or “&lt;bracketed Name&gt;’s father, [OtherName]”),
              produce (Name, has_father, OtherName) and (OtherName, has_daughter, Name).
            </rule>

            <rule name="Naming-a-daughter">
              Whenever you see “&lt;bracketed Parent&gt; had a daughter named [Child]”,
              infer (Parent, has_daughter, Child) and (Child, has_mother, Parent).
            </rule>

            <rule name="Sister-sibling">
              Whenever you see “[Name1] and his sister [Name2] …” or “[Name1] and her sister [Name2] …”,
              infer (Name1, has_sister, Name2) and (Name2, has_brother, Name1) if Name1 is male,
              or (Name1, has_sister, Name2) and (Name2, has_sister, Name1) if both are female.
              Otherwise, do not infer a sibling link.
            </rule>

            <rule name="Uncle">
              Whenever you see “&lt;bracketed NieceOrNephew&gt; went to her uncle [OtherName]’s house” (or any similar phrasing),
              infer (NieceOrNephew, has_uncle, OtherName) and (OtherName, has_niece, NieceOrNephew).
            </rule>

            <rule name="Shared-Parent Siblings">
              Whenever you see two bracketed names appear together and share a parent phrase:
                • “[A] and [B] asked their mother [C] …”
                • “[A] and [B] went to see their father [C] …”
              then:
                1. If [C] is explicitly named in brackets, infer each child’s parent link:
                   (A, has_mother, C) / (B, has_mother, C) if it’s “their mother [C]”,
                   or (A, has_father, C) / (B, has_father, C) if it’s “their father [C]”.
                2. Because A and B share the same named parent [C], infer they are siblings:
                   – If A is male and B is female: (A, has_sister, B) and (B, has_brother, A).
                   – If both are male: (A, has_brother, B) and (B, has_brother, A).
                   – If both are female: (A, has_sister, B) and (B, has_sister, A).
                   – If one’s gender cannot be determined, do not infer a sibling triple.
              (If the shared parent [C] is not named in brackets, do not create the parent link—only infer siblings if there’s a clear pronoun “their mother/father.”)
            </rule>

            <rule name="Grandson-naming">
              Whenever you see “&lt;bracketed Name&gt; has … grandson. The grandson’s name is [OtherName]”,
              infer (Name, has_grandson, OtherName) and (OtherName, has_grandfather, Name).
            </rule>

            <rule name="Other-explicit-relationships">
              Identify all other explicitly stated or strongly implied family relationships—this includes:
              • Direct parent/child (has_mother / has_father / has_son / has_daughter) only if none of the above rules applies.
              • Sibling (has_brother / has_sister) only if the text literally says “brother” or “sister,” or via the sibling rules above.
              • Aunt/uncle/niece/nephew.
              • Grandparent/grandchild (has_grandmother / has_grandfather / has_grandson / has_granddaughter) only if not already covered by “Grandson-naming.”
            </rule>

            <rule name="No-invent">
              Do not invent any new names or assume any relationships beyond what is explicitly stated or clearly implied.
            </rule>
          </behavior>

          <format>
            1. First, provide a detailed, step-by-step reasoning showing how you:
               • Resolved each special phrase (e.g., “X’s father, [Y]”, “had a daughter named [Y]”, “A and his sister, [B]”, “has … grandson. The grandson’s name is [Y]”, etc.).
               • Determined gender when pronouns like “her” or “his” appeared (or used name-based inference if no pronoun).
               • Decided which rule applied (Spouse-detection, Mother’s-husband, Father’s, Son-naming, Dad/Father-naming, Naming-a-daughter, Sister-sibling, Uncle, Shared-Parent Siblings, Grandson-naming, Other-explicit-relationships, etc.).
               • Extracted each family link and mapped it to exactly one relation from the allowed list.

            2. Then, print exactly this heading on its own line (no extra text):
               ### TRIPLES_START

            3. Below that line, list every relationship as exactly two triples—one direct and one inverse—using only the bracketed names and parentheses. Do NOT use any quotes or square brackets. Each triple must follow this format:
               (Subject, relationship, Object)
               And “relationship” must come from this list (exactly as written, with underscores):
               {self.dataset_config['relationships_list']}

            4. Finally, after listing all triples, end without adding any extra text or parentheses.
          </format>
        </system>
        """),
            ("human", "{statement}"),
        ])

        llm_model = get_llm(
            llm_type=self.general_config["llm_type"],
            llm_model=self.general_config["llm_model"],
            api_key=self.openai_api_key
        )
        # chain = prompt | llm_model.with_structured_output(schema=KnowledgeGraph)
        if self.general_config["llm_type"] != "ollama":
            chain = prompt | llm_model | StrOutputParser()
        else:
            chain = llm_model
        return chain

    def build_pre_revised_answers_generator_chain(self):
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

    def build_revised_answers_generator_chain(self):
        prompt = ChatPromptTemplate.from_template(
            self.get_revision_prompt(
                self.dataset_name,
                self.dataset_config['entity_classes_list']
            )
        )
        llm_model = get_llm(
            llm_type=self.general_config["llm_type"],
            llm_model=self.general_config["llm_model"],
            api_key=self.openai_api_key
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
        knowledge_graph_str_extractor_chain = self.build_knowledge_graph_str_extractor_chain()
        pre_revised_answers_generator_chain = self.build_pre_revised_answers_generator_chain()
        # revised_answers_generator_chain = self.build_revised_answers_generator_chain()

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

                    for graph_str in graphs_str_list:
                        print(f"Graph: {graph_str}")
                        graphs_list.append(KnowledgeGraphUtils.extract_triples_from_llm_str_output(graph_str))
                        print("-----------------------------------------------------")

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
