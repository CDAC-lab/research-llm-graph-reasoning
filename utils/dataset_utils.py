import ast
import pandas as pd


class DatasetUtils:

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    @staticmethod
    def pre_process_clutrr_dataset(dataset):
        # filter out test dataset
        df_test = dataset['test'].to_pandas()
        df_test['story_edges_list'] = df_test['story_edges'].apply(ast.literal_eval)
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
    def pre_process_clutrr_chains_for_prompt(graph_df):
        """
        Pre-process the output of the knowledge graph extractor to prepare it for the final answer generator.
        """
        questions_dict_list = []

        for index, row in graph_df.iterrows():
            relation_chain = row["chain"]
            parts = relation_chain.split("->")
            first = parts[0].split("(")[0].strip()
            last = parts[-1].strip()

            questions_dict_list.append({
                "relation_chain": relation_chain,
                "first": first,
                "last": last
            })

        return questions_dict_list