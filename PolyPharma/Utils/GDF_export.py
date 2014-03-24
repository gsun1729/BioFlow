__author__ = 'ank'

import numpy as np
from PolyPharma.configs import Dumps

Authorised_names = ['VARCHAR', 'DOUBLE', 'BOOLEAN', 'DOUBLE']

class GDF_export_Interface(object):
    """

    :param target_fname: name of the file to which the GDF file will be written to
    :param field_names: Names of different fields for the node description
    :param field_types: Types of different nodes for the node description
    :param node_properties_dict: dictionary mapping the node labels to outputs
    :param mincurrent: minimal current below which we are not rendering the links anymore
    :param Idx2Label: Mapping from the indexes curent matrix lines/columns to the node labels
    :param current_Matrix: matrix of currents from which we wish to rendred the GDF
    """

    def __init__(self, target_fname, field_names, field_types, node_properties_dict, mincurrent, Idx2Label, current_Matrix):
        self.target_file = open(target_fname, 'w')
        self.field_types = field_types
        self.field_names = field_names
        self.node_properties = node_properties_dict
        self.Idx2Label = Idx2Label
        self.mincurrent = mincurrent # minimal current for which we will be performing filtering out of the conductances and nodes through whichthe trafic is below that limit
        self.current_Matrix = current_Matrix # matrix where M[i,j] = current intesitu from i to j. Triangular superior, if current is from j to i, current is negative
        # current retrieval for the output should be done by getting all the non-zero terms of the current matrix and then filtering out terms/lines that have too little absolute current
        # rebuilding a new current Matrix and creating a dict to map the relations from the previous matrix into a new one.
        self.verify()


    def verify(self):
        """
        :raises Exception: "GDF Node declaration is wrong!" - of the length of field names and field type differ
        :raises Exception: "Wrong Types were declared, ...." - if the declared types are not in the Authorised names
        """
        if len(self.field_types) != len(self.field_types):
            raise Exception('GDF Node declaration is wrong')
        if not set(Authorised_names) >= set(self.field_types):
            raise  Exception('Wrong types were declared. please refer to the module for authorized types list')


    def write_nodedefs(self):
        """
        Takes in the dictionary that maps and returns the nodedefs line

        """
        accumulator = []
        for name, type in zip(self.field_names, self.field_types):
            accumulator.append(name+' '+type)
        retstring = ', '.join(accumulator)
        retstring = 'nodedef> name VARCHAR, current DOUBLE, '+retstring+'\n'
        self.target_file.write(retstring)


    def write_nodes(self):
        """
        Write the nodes with associated informations

        """
        for nodename, nodeprops in self.node_properties.iteritems():
            self.target_file.write(nodename +', '+', '.join(nodeprops)+'\n')


    def write_edgedefs(self):
        """
        Write defintions for the edges. Right now, the information passing through the edges is restricted to the current

        """
        retstring = 'edgedef> node1 VARCHAR, node1 VARCHAR, weight DOUBLE\n'
        self.target_file.write(retstring)


    def write_edges(self):
        """
        Writes informations about edges connections. These informations are pulled from the conductance matrix.

        """
        nz = self.current_Matrix.nonzero()
        for i, j in zip(nz[0], nz[1]):
            if abs(self.current_Matrix[i,j]) > self.mincurrent:
                self.target_file.write(self.Idx2Label[i]+', '+self.Idx2Label[j]+', '+str(self.current_Matrix[i,j])+'\n')
        self.target_file.close()


if __name__ == "__main__":

    premat = np.zeros((3,3))
    premat[0,1] = 1.0
    premat[0,2] = 4.0
    premat[1,2] = 0.5
    GDFW = GDF_export_Interface(Dumps.GDF_debug, ['test'],['VARCHAR'],
                                {'test1':['test one'], 'test2':['test two'], 'test3':['test three']},
                                0.0, {0:'test1',1:'test2', 2: 'test3'}, premat)
    GDFW.write_nodedefs()
    GDFW.write_nodes()
    GDFW.write_edgedefs()
    GDFW.write_edges()
