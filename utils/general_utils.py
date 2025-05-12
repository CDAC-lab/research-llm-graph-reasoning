import pandas as pd
import os

class GeneralUtils:

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def save_pre_revised_answers_as_csv(self, pre_revised_answers_list, chain_df, batch_num):
        """
        Save the final answers to a CSV file.
        """
        final_answers_list_str = [x.answer for x in pre_revised_answers_list]
        reasons_list_str = [x.reason for x in pre_revised_answers_list]
        chain_df["final_answer"] = final_answers_list_str
        chain_df["reason"] = reasons_list_str
        output_file_path = f"./outputs/pre_revised_answers/{self.dataset_name}_pre_revised_answers_b{batch_num}.csv"
        chain_df.to_csv(output_file_path, index=False)
        return output_file_path

    def save_revised_answers_as_csv(self, revised_answers_list, chain_df, batch_num):
        """
        Save the final answers to a CSV file.
        """
        revised_answers_list_str = [x.revised_answer for x in revised_answers_list]
        revised_reasons_list_str = [x.reason for x in revised_answers_list]
        previous_reasoning_errors_list_str = [x.previous_reasoning_errors for x in revised_answers_list]
        chain_df["revised_answer"] = revised_answers_list_str
        chain_df["revised_reason"] = revised_reasons_list_str
        chain_df["previous_reasoning_errors"] = previous_reasoning_errors_list_str
        output_file_path = f"./outputs/revised_answers/{self.dataset_name}_revised_answers_b{batch_num}.csv"
        chain_df.to_csv(output_file_path, index=False)
        return output_file_path

    # def save_conventional_answers_as_csv(self, conventional_answers_list, dicts_chunk, batch_num):
    #     """
    #     Save the final answers to a CSV file.
    #     """
    #     conventional_answers_list_str = [x.answer for x in conventional_answers_list]
    #     reasons_list_str = [x.reason for x in conventional_answers_list]
    #
    #     for i in range(len(dicts_chunk)):
    #         dicts_chunk[i]["question_num"] = i
    #
    #     dicts_chunk_df = pd.DataFrame(dicts_chunk)
    #     output_file_path = f"./outputs/conventional_answers/{self.dataset_name}_conventional_answers_b{batch_num}.csv"
    #     df = pd.DataFrame({
    #         "answer": conventional_answers_list_str,
    #         "reason": reasons_list_str
    #     })
    #     df["batch_num"] = batch_num
    #     df["question_num"] = dicts_chunk_df["question_num"]
    #     df["num_hops"] = dicts_chunk_df["num_hops"]
    #     df["target_text"] = dicts_chunk_df["target_text"]
    #     df["query"] = dicts_chunk_df["query"]
    #     df.to_csv(output_file_path, index=False)
    #     return output_file_path

    def save_conventional_answers_as_csv(self, conventional_answers_list, dicts_chunk, batch_num):
        """
        Save the final answers to a CSV file.
        """
        conventional_answers_list_str = [x for x in conventional_answers_list]

        for i in range(len(dicts_chunk)):
            dicts_chunk[i]["question_num"] = i

        dicts_chunk_df = pd.DataFrame(dicts_chunk)
        output_file_path = f"./outputs/conventional_answers/{self.dataset_name}_conventional_answers_b{batch_num}.csv"
        df = pd.DataFrame({
            "answer": conventional_answers_list_str
        })
        df["batch_num"] = batch_num
        df["question_num"] = dicts_chunk_df["question_num"]
        df["num_hops"] = dicts_chunk_df["num_hops"]
        df["target_text"] = dicts_chunk_df["target_text"]
        df["query"] = dicts_chunk_df["query"]
        df.to_csv(output_file_path, index=False)
        return output_file_path

    def filter_shortest_paths(self, batch_num):
        """
        Filter the shortest paths based on the minimum chain length per (batch_num, question_num) group.
        """
        input_path = f"./outputs/query_outputs/{self.dataset_name}_query_outputs_b{batch_num}.csv"
        output_path = f"./outputs/shortest_paths/{self.dataset_name}_shortest_paths_b{batch_num}.csv"

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"❌ Input file not found: {input_path}")

        try:
            df = pd.read_csv(input_path)
        except Exception as e:
            raise IOError(f"❌ Failed to read CSV file: {input_path}\nError: {e}")

        required_columns = {'chain', 'target_text', 'num_hops', 'batch_num', 'question_num'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"❌ Missing required columns in CSV: {missing}")

        def get_shortest_chain_row(group):
            group = group.copy()
            # Ensure 'chain' is a string and handle missing/null values
            group['chain'] = group['chain'].astype(str).fillna("")
            group['chain_length'] = group['chain'].apply(len)
            return group.loc[group['chain_length'].idxmin(), ['chain', 'target_text', 'num_hops']]

        try:
            result_df = df.groupby(['batch_num', 'question_num']).apply(get_shortest_chain_row).reset_index()
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result_df.to_csv(output_path, index=False)
            print(f"✅ Saved {len(result_df)} rows to {output_path}")
            return output_path
        except Exception as e:
            raise RuntimeError(f"❌ Error during the shortest path processing or saving results: {e}")


