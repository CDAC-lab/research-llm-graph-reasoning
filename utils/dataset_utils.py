import ast
import re
import pandas as pd


class DatasetUtils:

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    @staticmethod
    def pre_process_clutrr_dataset(dataset, is_debug, sample_question_indexes):
        # filter out test dataset
        df_test = dataset['test'].to_pandas()
        df_test['story_edges_list'] = df_test['story_edges'].apply(ast.literal_eval)

        if is_debug:
            df_test = df_test.iloc[sample_question_indexes]

        list_dict_test = df_test[['story', 'query', 'target_text', 'story_edges_list']].to_dict(orient='records')

        list_questions = []
        for i in range(len(list_dict_test)):
            list_questions.append({
                "statement": list_dict_test[i]['story']
            })
            list_dict_test[i]['num_hops'] = len(list_dict_test[i]['story_edges_list'])
            list_dict_test[i].pop('story', None)
            list_dict_test[i].pop('story_edges_list', None)

        del dataset
        del df_test

        return list_dict_test, list_questions

    @staticmethod
    def pre_process_pre_revised_questions_for_prompt(graph_df):
        """
        Pre-process the output of the knowledge graph extractor to prepare it for the final answer generator.
        """
        questions_dict_list = []

        for index, row in graph_df.iterrows():
            relation_chain = row["chain"]
            relation_str = DatasetUtils.chain_to_triples(relation_chain)
            parts = relation_chain.split("->")
            first = parts[0].split("(")[0].strip()
            last = parts[-1].strip()

            questions_dict_list.append({
                "relation_str": relation_str,
                "first_person": first,
                "last_person": last
            })

        return questions_dict_list

    @staticmethod
    def chain_to_triples(chain: str) -> str:
        """
        Converts a relationship chain like:
            "Ashley (has_grandson) -> Tony (has_sister) -> Charlotte"
        Into a list of triples:
            [("Ashley", "has_grandson", "Tony"), ("Tony", "has_sister", "Charlotte")]
        """
        # Split the chain into parts by '->'
        parts = [part.strip() for part in chain.split('->')]

        nodes = []
        relations = []

        for part in parts:
            match = re.match(r'^(\w+)(?:\s*\((\w+)\))?$', part)
            if match:
                name, relation = match.groups()
                nodes.append(name)
                if relation:
                    relations.append(relation)
            else:
                raise ValueError(f"Invalid part format: {part}")

        # Sanity check
        if len(nodes) != len(relations) + 1:
            raise ValueError("The number of relationships must be one less than the number of entities.")

        # Build triples
        triples = [
            (nodes[i], relations[i], nodes[i + 1])
            for i in range(len(relations))
        ]

        return_str = ""
        for i, (subject, relation, obj) in enumerate(triples):
            return_str += f"{i + 1}. ({subject}, {relation}, {obj})" + " \n"

        return return_str

    @staticmethod
    def pre_process_revised_questions_for_prompt(graph_df):
        """
        Pre-process the output of the knowledge graph extractor to prepare it for the final answer generator.
        """
        questions_dict_list = []

        for index, row in graph_df.iterrows():
            relation_chain = row["chain"]
            relation_str = DatasetUtils.chain_to_triples(relation_chain)

            parts = relation_chain.split("->")
            first = parts[0].split("(")[0].strip()
            last = parts[-1].strip()

            questions_dict_list.append({
                "relation_str": relation_str,
                "selected_answer": row["final_answer"],
                "first_person": first,
                "last_person": last,
                "reasoning_steps": row["reason"]
            })

        return questions_dict_list

    @staticmethod
    def conventional_questions_for_prompt(questions_chunk, dicts_chunk):
        """
        Pre-process the output of the knowledge graph extractor to prepare it for the final answer generator.
        """
        questions_dict_list = []

        for i in range(len(questions_chunk)):
            input_story = questions_chunk[i]['statement']
            query = ast.literal_eval(dicts_chunk[i]['query'])

            questions_dict_list.append({
                "input_story": input_story,
                "question": f"({query[0]}, ?, {query[-1]})"
            })

        return questions_dict_list
