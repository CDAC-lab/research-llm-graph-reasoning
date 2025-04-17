import pandas as pd

class GeneralUtils:

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def save_final_answer_as_csv(self, final_answers_list, chain_df, batch_num):
        """
        Save the final answers to a CSV file.
        """
        final_answers_list_str = [x.answer for x in final_answers_list]
        chain_df["final_answer"] = final_answers_list_str
        chain_df.to_csv(f"./outputs/final_answers/{self.dataset_name}_final_answers_b{batch_num}.csv", index=False)