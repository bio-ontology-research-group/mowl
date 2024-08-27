import os
import time
from deprecated.sphinx import versionchanged, versionadded
import tempfile


class WalkingModel():

    '''
    Base class for walking methods.

    :param num_walks: Number of walks per node
    :type num_walks: int
    :param walk_length: Length of each walk
    :type walk_length: int
    :param workers: Number of threads to be used for computing the walks, defaults to 1'
    :type workers: int, optional
    '''

    def __init__(self, num_walks, walk_length, outfile, workers=1):

        if not isinstance(num_walks, int):
            raise TypeError("Parameter num_walks must be an integer")
        if not isinstance(walk_length, int):
            raise TypeError("Parameter walk_length must be an integer")
        if not isinstance(workers, int):
            raise TypeError("Optional parameter workers must be an integer")

        if outfile is None:
            tmp_file = tempfile.NamedTemporaryFile()
            self.outfile = tmp_file.name
        else:
            if not isinstance(outfile, str):
                raise TypeError("Optional parameter outfile must be a string")
            self.outfile = outfile

        self.num_walks = num_walks
        self.walk_length = walk_length
        self.workers = workers

    # Abstract methods
    @versionchanged(version="0.1.0", reason="The method now can accept a list of entities to \
        focus on when generating the random walks.")
    def walk(self, edges, nodes_of_interest=None):
        '''
        This method will generate random walks from a graph in the form of edgelist.

        :param edges: List of edges
        :type edges: :class:`mowl.projection.edge.Edge`
        :param nodes_of_interest: List of entity names to filter the generated walks. If a walk \
        contains at least one word of interest, it will be saved into disk, otherwise it will be \
        ignored.  If no list is input, all the nodes will be considered. Defaults to ``None``
        :type nodes_of_interest: list, optional
        '''

        raise NotImplementedError()


    def wait_for_all_walks(self):
        """
        This method waits until all the walks are written to the output file.
        """
        cooldown_period = 1
        stable_since = None
        last_modified = os.path.getmtime(self.outfile)
        while True:
            time.sleep(0.1)
            current_modified = os.path.getmtime(self.outfile)
            if current_modified == last_modified:
                if stable_since is None:
                    stable_since = time.time()
                elif time.time() - stable_since > cooldown_period:
                    break

            else:
                stable_since = None

            last_modified = current_modified

        return
