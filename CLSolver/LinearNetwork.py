import pickle
import logging
import numpy as np
import networkx as nx
from typing import Union, List, Optional
from pathlib import Path

class LinearNetworkError(Exception):
    """Custom exception for errors specific to the LinearNetwork class."""
    pass

class LinearNetwork:
    """
    A class representing a Linear Network, capable of initializing from different sources:
    a pickle file containing a graph dictionary, a pickle file with a NetworkX graph object,
    directly from a NetworkX graph object, or uninitialized.

    Attributes:
        is_initialized (bool): Flag indicating whether the network has been properly initialized.
        NN (int): Number of nodes in the graph.
        NE (int): Number of edges in the graph.
        EI (List[int]): List of start nodes for each edge.
        EJ (List[int]): List of end nodes for each edge.
    """

    def __init__(self, input_data: Optional[Union[str, nx.Graph]] = None) -> None:
        """
        Initialize the object with graph data.

        Args:
            input_data: Path to a pickle file or a networkx Graph object. If None,
                        initializes without loading graph data.

        Raises:
            ValueError: If input_data is not None, str, or nx.Graph.
        """
        self._is_initialized: bool = False
        self._NN: int = 0
        self._NE: int = 0
        self._EI: np.ndarray = np.array([], dtype=int)
        self._EJ: np.ndarray = np.array([], dtype=int)

        if isinstance(input_data, str):
            self.load_from_pickle(input_data)
        elif isinstance(input_data, nx.Graph):
            self.extract_graph_properties_from_networkx(input_data)
        elif isinstance(input_data, dict):
            self.initialize_graph_dict(input_data)
        elif input_data is not None:
            raise ValueError("Unsupported type for input_data")

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @is_initialized.setter
    def is_initialized(self, value: bool) -> None:
        self._is_initialized = value

    @property
    def NN(self) -> int:
        return self._NN

    @NN.setter
    def NN(self, value: int) -> None:
        self._NN = value

    @property
    def NE(self) -> int:
        return self._NE

    @NE.setter
    def NE(self, value: int) -> None:
        self._NE = value

    @property
    def EI(self) -> np.ndarray:
        return self._EI

    @EI.setter
    def EI(self, value: Union[List[int], np.ndarray]) -> None:
        self._EI = np.array(value, dtype=int)

    @property
    def EJ(self) -> np.ndarray:
        return self._EJ

    @EJ.setter
    def EJ(self, value: Union[List[int], np.ndarray]) -> None:
        self._EJ = np.array(value, dtype=int)

    def load_from_pickle(self, pickle_path: str) -> None:
        """
        Loads the network data from a pickle file. The file can contain either a graph dictionary
        or a NetworkX graph object.

        Args:
            pickle_path (str): The path to the pickle file.

        Raises:
            FileNotFoundError: If no file is found at the specified path.
            LinearNetworkError: If the file content is neither a dict nor a NetworkX graph.
        """
        if not Path(pickle_path).exists():
            raise FileNotFoundError(f"No pickle file found at path: {pickle_path}")

        try:
            with open(pickle_path, 'rb') as f:
                loaded_data = pickle.load(f)
                if isinstance(loaded_data, dict):
                    self.initialize_graph_dict(loaded_data)
                elif isinstance(loaded_data, nx.Graph):
                    self.extract_graph_properties_from_networkx(loaded_data)
                else:
                    raise LinearNetworkError("Pickle file content is neither a dict nor a NetworkX graph.")
        except Exception as e:
            logging.error(f"Failed to load and process pickle file: {e}")
            raise LinearNetworkError(f"Failed to process pickle file: {e}")

    def initialize_graph_dict(self, graph_dict: dict) -> None:
        """
        Initializes the network from a dictionary containing graph properties.

        Args:
            graph_dict (dict): A dictionary containing graph properties with keys 'NN', 'NE', 'EI', and 'EJ'.
        """
        try:
            self._NN = graph_dict['NN']
            self._NE = graph_dict['NE']
            self._EI = graph_dict['EI']
            self._EJ = graph_dict['EJ']
        except KeyError as e:
            logging.error(f"Missing key in graph_dict: {e}")
            raise ValueError(f"Required graph property missing: {e}") from None
        else:
            self._is_initialized = True

    def extract_graph_properties_from_networkx(self, graph: nx.Graph) -> None:
        """
        Extracts graph properties from a NetworkX graph object and initializes the network.

        Args:
            graph (nx.Graph): A NetworkX graph object from which to extract properties.
        """
        self._NN = graph.number_of_nodes()
        self._NE = graph.number_of_edges()
        self._EI = [edge[0] for edge in graph.edges()]
        self._EJ = [edge[1] for edge in graph.edges()]
        self._is_initialized = True

    def to_networkx_graph(self) -> nx.Graph:
        """
        Constructs and returns a NetworkX graph based on the current attributes (NN, NE, EI, EJ).

        Returns:
            nx.Graph: A NetworkX graph constructed from the class attributes.

        Raises:
            LinearNetworkError: If the class attributes are not properly initialized.
        """
        if not self._is_initialized:
            raise LinearNetworkError("LinearNetwork class is not initialized with graph data.")

        G = nx.Graph()

        # Node indices are assumed to start from 0 and be continuous.
        G.add_nodes_from(range(self._NN))

        # Add edges
        for start_node, end_node in zip(self._EI, self._EJ):
            if start_node >= self._NN or end_node >= self._NN:
                raise LinearNetworkError("Edge node index out of bounds.")
            G.add_edge(start_node, end_node)

        return G
