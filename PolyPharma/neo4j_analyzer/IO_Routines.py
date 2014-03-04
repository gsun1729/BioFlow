__author__ = 'ank'

from PolyPharma.neo4j_Declarations.Graph_Declarator import DatabaseGraph
from PolyPharma.configs import IDFilter, edge_type_filters

def lookup_by_ID(Domain, req):
    """
    Looks up a node by legacy ID and prints it's access url. The lookup is a case-sensitive strict match.
    :param Domain: Node type of the form of DatabaseGraph.Object
    :param req: requested legacy ID
    """
    retset = Domain.index.lookup( ID = req )
    if not retset:
        print "nothing found"
    if retset:
        for item in retset:
            print item


def count_items(Domain):
    i = 0
    for elt in Domain.get_all():
        i += 1
    return i


def Look_up_by_ID_for_a_set(Domain, ID_set):
    """
    Looks up nodes by legacy IDs from an ID_set list and prints their access url. The lookup is a case-sensitive strict match.
    :param Domain: Node type of the form of DatabaseGraph.Object
    :param Domain: Node type of the form of DatabaseGraph.Object
    :param ID_set: list of requested legacy ID
    """
    for ID in ID_set:
        print "scanning for:", ID
        lookup_by_ID(Domain, ID)
        print "=================="


def Look_up_Annot_Node(p_load, p_type = ''):
    """
    Looks up nodes accessible via the annotation nodes with a given annotation and given annotation type.
    The lookup strict match, but case-insensitive.

    .. code-block: python
    >>> print Look_up_Annot_Node('ENSG00000131981', 'UNIPROT_Ensembl')
    >>> # TODO: add the results here when implementing the doctests

    :param p_load: payload
    :param p_type: payload type
    :return: node type, node's displayName, node's db_ID, node's legacy ID
    :rtype: 4- tuple
    :raise Exception: "p_type unsupported", in case a p_type is not on the supported list specified in the neo4j_Declarations.neo4j_typeDec
    """

    def run_through(node_generator):
        """
        Supporting function. Gets the nodes that are referenced by the annot_nodes in the generator
        :param node_generator: iterator over annot_nodes
        :return: the list of object nodes accessible from this set of annot_nodes
        :raise Warning: if an annotation node is not bound to a real node. This might happen if some object nodes were manually
                        deleted, but not their annotation nodes. Tu curb this a full database reload is required
        """
        retset = []
        for node in node_generator:
            gen_2 = node.inV("is_annotated")
            if not node_generator:
                raise Warning(str(node) + "is floating alone in the wild. He feels lonely.")
            for rel_node in gen_2:
                node_db_ID = str(rel_node).split('/')[-1][:-1]
                node_ID = rel_node.ID
                node_type = rel_node.element_type
                node_display = rel_node.displayName
                retset.append((node_type, node_display, node_db_ID, node_ID))

        return retset

    def double_index_search(pload, ptype):
        """
        Supporting fucntion. Performs a search in the annotation nodes over both a payload and a ptype
        :param pload: payload
        :param ptype: payload type
        :return: list of found annotation nodes satisfying both payload content and payload type conditions
        """
        node_generator = DatabaseGraph.AnnotNode.index.lookup(payload = pload)
        retset = []
        if node_generator:
            for node in node_generator:
                if ptype in node.ptype:
                    retset.append(node)
        return retset

    from PolyPharma.neo4j_Declarations.neo4j_typeDec import Anot_Node_ptypes
    pload = p_load.upper()
    if p_type == '':
        node_generator = DatabaseGraph.AnnotNode.index.lookup(payload = pload)
        if not node_generator:
            return []
        return run_through(node_generator)

    if p_type in Anot_Node_ptypes:
        node_generator =  double_index_search(pload, p_type)
        if not node_generator:
            return []
        return run_through(node_generator)

    raise Exception(p_type + "is unsupported. Please refer to Anot_Node_ptypes in neo4j_typeDec for supported types")


def Erase_custom_fields():
    """
        Resets the .costum field of all the Nodes on which we have iterated here. Usefull to perform
        after node set or node connectivity were modfied.

        Unlike the method in the Matrix_retrieval cluster, this method is very time-consuming, since it iterates
        on all the elements of all the classes susceptible to have the costum field.
    """

    # TODO: reconfigure to erase connexity infos later on.
    Node_gen = DatabaseGraph.Node.get(costum = 'Main_Connex')

    for Node in Node_gen:
        Node.custom = ''
        Node.save()


def reaction_participant_getter(Reaction, main_connex_only):
    """
    Recovers all the participants of the reaction
    :param Reaction: Reaction node for which we are willing to get the participants
    :param main_connex_only: If set to true, will only pull elements from the reaction that are in the main connex_set
    :type main_connex_only: bool
    :return: List of found nodes, number of found nodes
    :rtype: 2- tuple
    """
    edge_type_filter = edge_type_filters["Reaction"]
    LocalList = []
    count = 0
    for edge_type in edge_type_filter:
        if Reaction.bothV(edge_type) == None:
            continue
        for elt in Reaction.bothV(edge_type):
            Connex = True

            if main_connex_only:
                Connex = False
                if  elt.custom != None and "Main_Connex" in elt.custom:
                    Connex = True

            ID = str(elt).split('/')[-1][:-1]
            if ID not in IDFilter and Connex:
                LocalList.append(ID)
                count += 1
    return LocalList, count


def expand_from_seed(Seed_Node_ID, edge_filter):
    """
    Recovers all the nodes accessible in one jump from a seed_node with a given database ID by jumping only via the relations
        of type specified in the edge_filter
    :param Seed_Node_ID: the database ID of the initial node from which we are observing accessibility
    :param edge_filter: the list of relation types for which the jumps are authorised
    :return: List of found nodes database ID, number of found nodes
    :rtype: 2- tuple
    """
    Seed_Node = DatabaseGraph.vertices.get(Seed_Node_ID)
    LocalList = []
    count = 0
    for edge_type in edge_filter:
        if Seed_Node.bothV(edge_type) != None:
            for elt in Seed_Node.bothV(edge_type):
                ID = str(elt).split('/')[-1][:-1]
                if ID not in IDFilter:
                    LocalList.append(ID)
                    count += 1
    return LocalList, count



if __name__ == "__main__":
    # print count_items(DatabaseGraph.UNIPORT)
    # lookup_by_ID(DatabaseGraph.UNIPORT, "CK2N2_HUMAN")
    # Erase_custom_fields()

    print Look_up_Annot_Node('ENSG00000131981', 'UNIPROT_Ensembl')