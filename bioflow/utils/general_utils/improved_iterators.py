"""
Contains the tools required for a saner operation on iterables
"""


def unique_product(*seqs):
    """
    product of sequences with eliminated repeats.

    All credit goes to http://stackoverflow.com/users/2705542/tim-peters

    Code from

    http://stackoverflow.com/questions/19744542/itertools-product-eliminating-repeated-elements

    :param seqs: set of sequences used as a product
    :return:
    """
    def inner_iterator(i):
        if i == n:
            yield tuple(result)
            return

        for elt in sets[i] - seen:
            seen.add(elt)
            result[i] = elt
            for _tuple in inner_iterator(i+1):
                yield _tuple
            seen.remove(elt)

    sets = [set(seq) for seq in seqs]
    n = len(sets)
    seen = set()
    result = [None] * n
    for v_tuple in inner_iterator(0):
        yield v_tuple