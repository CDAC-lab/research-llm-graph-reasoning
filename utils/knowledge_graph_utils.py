from rdflib import Graph, Namespace, RDF, RDFS, OWL, URIRef
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import os
import ast

class KnowledgeGraphUtils:
    """
    A utility class for handling knowledge graph operations.
    """
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    @staticmethod
    def load_owl_graph(owl_file):
        """Loads an OWL file into an rdflib graph."""
        g = Graph()
        g.parse(owl_file, format="xml")  # Parse RDF/XML format
        return g

    @staticmethod
    def add_individuals(ontology_graph, individuals_with_classes, namespace):
        """
        Adds individuals to the ontology graph and assigns them to their respective classes.

        :param namespace: namespace of the ontology.
        :param ontology_graph: An RDFLib Graph object.
        :param individuals_with_classes: A dictionary where keys are individual IRIs and values are lists of class IRIs.
        """
        for individual, classes in individuals_with_classes.items():
            individual_iri = f"{namespace}#{individual}"
            individual_uri = URIRef(individual_iri)
            for cls in classes:
                class_iri = f"{namespace}#{cls}"
                ontology_graph.add((individual_uri, RDF.type, URIRef(class_iri)))

    @staticmethod
    def add_relationship(ontology_graph, entity_1, entity_2, object_property):
        """
        Adds a relationship between two existing individuals.

        :param ontology_graph: An RDFLib Graph object.
        :param entity_1: IRI of the first entity.
        :param entity_2: IRI of the second entity.
        :param object_property: IRI of the object property defining the relationship.
        """
        ontology_graph.add((URIRef(entity_1), URIRef(object_property), URIRef(entity_2)))

    def save_llm_response_as_owl(self, graphs_list, batch_num):
        i = 0
        for graph in graphs_list:
            # Create a new graph
            g = Graph()

            # Define the namespace
            ontology_name = f"{self.dataset_name}_ontology_b{batch_num}_q{i}"
            output_owl_file = f"./outputs/knowledge_graphs/{ontology_name}.rdf"
            namespace = f"http://www.semanticweb.org/sudheera/{self.dataset_name}/2025/2/{ontology_name}"

            # Add individuals and relationships to the graph
            for entity in graph.entity_classes:
                # for entity in test_entity_classes:
                self.add_individuals(g, {entity.entity: entity.classes}, namespace)

            for relationship in graph.graph:
                # for relationship in test_graph:
                entity_1 = f"{namespace}#{relationship.entity_1}"
                entity_2 = f"{namespace}#{relationship.entity_2}"
                object_property = f"{namespace}#{relationship.edge}"
                self.add_relationship(g, entity_1, entity_2, object_property)

            # Save the graph to an OWL file
            g.serialize(destination=output_owl_file, format="xml")
            i += 1

    @staticmethod
    def find_paths(graph, start, end, path=None, properties=None):
        if path is None:
            path = []
        if properties is None:
            properties = []

        # Add the current node to the path
        path.append(start)

        # If we reach the end, return the current path with its properties
        if start == end:
            return [(list(path), list(properties))]

        # List to hold all possible paths
        paths = []

        # Get all outgoing relationships from the current node
        for p, o in graph.predicate_objects(subject=start):
            # Avoid cycles: check if the node is already in the path
            if o not in path:
                # Add the property to the properties list
                new_properties = properties + [p]
                # Recursively search for paths from this node
                new_paths = KnowledgeGraphUtils.find_paths(graph, o, end, path.copy(), new_properties)
                for new_path, new_property in new_paths:
                    paths.append((new_path, new_property))

        return paths

    @staticmethod
    def query_knowledge_graph(file_path, question_dict, batch_num, q_index):
        """
        Query the knowledge graph to find relationships between individuals.
        """
        graph = Graph()
        graph.parse(file_path, format="xml")

        query_tuple = ast.literal_eval(question_dict['query'])

        start_individual = query_tuple[0]
        stop_individual = query_tuple[1]

        ns1_namespace = dict(graph.namespaces()).get("ns1")

        start_iri = URIRef(f"{ns1_namespace}{start_individual}")
        stop_iri = URIRef(f"{ns1_namespace}{stop_individual}")

        all_paths = KnowledgeGraphUtils.find_paths(graph, start_iri, stop_iri)
        # Return paths as a pandas df
        rows = []
        for i, (path, properties) in enumerate(all_paths, 1):
            # Format output: Show node and predicate (object property) in between
            path_str = " -> ".join([f"{str(path[j]).split('#')[-1]} ({str(properties[j]).split('#')[-1]})" if j < len(
                properties) else str(path[j]).split('#')[-1] for j in range(len(path))])
            rows.append({
                "chain": path_str,
                "query": query_tuple,
                "target_text": question_dict['target_text'],
                "num_hops": question_dict['num_hops'],
                "batch_num": batch_num,
                "question_num": q_index
            })

        return pd.DataFrame(rows)

    def query_knowledge_graph_batch(self, dicts_chunk, batch_num, max_workers):
        """
        Query the knowledge graph to find relationships between individuals.
        """
        futures = []
        dataframes = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for q_index, question_dict in enumerate(dicts_chunk):
                file_name = f"{self.dataset_name}_ontology_b{batch_num}_q{q_index}.rdf"
                file_path = os.path.join("./outputs/knowledge_graphs", file_name)

                if os.path.exists(file_path):
                    futures.append(
                        executor.submit(self.query_knowledge_graph, file_path, question_dict, batch_num, q_index)
                    )
                else:
                    print(f"⚠️ File not found: {file_path}")

        # Collect results
        for future in futures:
            try:
                df = future.result()
                if df is not None and not df.empty:
                    dataframes.append(df)
            except Exception as e:
                print(f"❌ Error processing: {e}")

        # Combine all into one DataFrame
        output_file = f"./outputs/query_outputs/{self.dataset_name}_query_outputs_b{batch_num}.csv"
        if dataframes:
            final_df = pd.concat(dataframes, ignore_index=True)
            final_df.to_csv(output_file, index=False)
            print(f"✅ Saved {len(final_df)} rows to {output_file}")
            return output_file
        else:
            print("⚠️ No data collected.")