import json
from dynamic_graph_building.dynamic_graph_builder import DynamicGraphBuilder


def run_clutrr_experiment():
    dataset_name = 'clutrr'
    dynamic_graph_builder = DynamicGraphBuilder(dataset_name)
    dynamic_graph_builder.execute()


if __name__ == '__main__':
    run_clutrr_experiment()
