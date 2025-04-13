from rdflib import Graph, Namespace, RDF, RDFS, OWL, URIRef

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

    def save_llm_response_as_owl(self, graphs_list):
        i = 0
        for graph in graphs_list:
            # Create a new graph
            g = Graph()

            # Define the namespace
            ontology_name = f"{self.dataset_name}_ontology_q{i}"
            output_owl_file = f"./outputs/knowledge_graphs/{ontology_name}.rdf"
            namespace = f"http://www.semanticweb.org/sudheera/{self.dataset_name}/2025/2/{ontology_name}"

            # Add individuals and relationships to the graph
            for entity in graphs_list[0].entity_classes:
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
