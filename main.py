import json
from dynamic_graph_building.dynamic_graph_builder import DynamicGraphBuilder
from conventional_prompting.conventional_executor import ConventionalExecutor


def run_clutrr_experiment(is_debug: bool, is_conventional: bool):
    dataset_name = 'clutrr'
    debug_question_indexes = [
        0, 2, 5, 7, 1, 12, 39, 42, 40, 41, 140, 38, 46, 146, 147, 148, 149, 166, 184, 145, 150, 223, 224, 226, 227, 242,
        347, 222, 225, 502, 510, 500, 503, 505, 450, 501, 507, 514, 524, 513, 518, 512, 523, 517, 526, 671, 674, 668,
        669, 670, 711, 667, 673, 804, 805, 803, 807, 827, 852, 806, 809, 935, 938, 928, 932, 927, 942, 926, 931
    ]
    if is_conventional:
        conventional_executor = ConventionalExecutor(
            dataset_name=dataset_name,
            is_debug=is_debug,
            sample_question_indexes=debug_question_indexes
        )
        conventional_executor.execute()
    else:
        dynamic_graph_builder = DynamicGraphBuilder(
            dataset_name=dataset_name,
            is_debug=is_debug,
            sample_question_indexes=debug_question_indexes
        )
        dynamic_graph_builder.execute()


if __name__ == '__main__':
    is_debug = True
    is_conventional = True
    run_clutrr_experiment(is_debug, is_conventional)
