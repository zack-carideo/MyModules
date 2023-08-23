import csv
import os
import time
from typing import List, Tuple, Callable, Any, Dict, Optional
import logging
import sys

import faiss
import numpy as np
import psutil
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt


class ScalableSemanticSearch:
    """Vector similarity using product quantization with sentence transformers embeddings and cosine similarity."""

    def __init__(self, device="cpu", model: SentenceTransformer =  None):
        self.device = device
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2" , device=self.device) if model is None else model
        self.dimension = self.model.shape[1] #self.model.get_sentence_embedding_dimension()
        self.quantizer = None
        self.index = None
        self.hashmap_index_sentence = None

        log_directory = "log"
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)
        log_file_path = os.path.join(log_directory, "scalable_semantic_search.log")

        logging.basicConfig(
            filename=log_file_path,
            level=logging.INFO,
            format="%(asctime)s %(levelname)s: %(message)s",
        )
        logging.info("ScalableSemanticSearch initialized with device: %s", self.device)

    @staticmethod
    def calculate_clusters(n_data_points: int) -> int:
        return max(2, min(n_data_points, int(np.sqrt(n_data_points))))

    def encode(self, data: List[str]) -> np.ndarray:
        """Encode input data using sentence transformer model.

        Args:
            data: List of input sentences.

        Returns:
            Numpy array of encoded sentences.
        """
        embeddings = self.model.encode(data)
        self.hashmap_index_sentence = self.index_to_sentence_map(data)
        return embeddings.astype("float32")

    def build_index(self, embeddings: np.ndarray) -> None:
        """Build the index for FAISS search.

        Args:
            embeddings: Numpy array of encoded sentences.
        """
        n_data_points = len(embeddings)
        if (
            n_data_points >= 1500
        ):  # Adjust this value based on the minimum number of data points required for IndexIVFPQ
            self.quantizer = faiss.IndexFlatL2(self.dimension)
            n_clusters = self.calculate_clusters(n_data_points)
            self.index = faiss.IndexIVFPQ(
                self.quantizer, self.dimension, n_clusters, 8, 4
            )
            logging.info("IndexIVFPQ created with %d clusters", n_clusters)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            logging.info("IndexFlatL2 created")

        if isinstance(self.index, faiss.IndexIVFPQ):
            self.index.train(embeddings)
        self.index.add(embeddings)
        logging.info("Index built on device: %s", self.device)

    @staticmethod
    def index_to_sentence_map(data: List[str]) -> Dict[int, str]:
        """Create a mapping between index and sentence.

        Args:
            data: List of sentences.

        Returns:
            Dictionary mapping index to the corresponding sentence.
        """
        return {index: sentence for index, sentence in enumerate(data)}

    @staticmethod
    def get_top_sentences(
        index_map: Dict[int, str], top_indices: np.ndarray
    ) -> List[str]:
        """Get the top sentences based on the indices.

        Args:
            index_map: Dictionary mapping index to the corresponding sentence.
            top_indices: Numpy array of top indices.

        Returns:
            List of top sentences.
        """
        return [index_map[i] for i in top_indices]

    def search(self, input_sentence: str, top: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute cosine similarity between an input sentence and a collection of sentence embeddings.

        Args:
            input_sentence: The input sentence to compute similarity against.
            top: The number of results to return.

        Returns:
            A tuple containing two numpy arrays. The first array contains the cosine similarities between the input
            sentence and the embeddings, ordered in descending order. The second array contains the indices of the
            corresponding embeddings in the original array, also ordered by descending similarity.
        """
        vectorized_input = self.model.encode(
            [input_sentence], device=self.device
        ).astype("float32")
        D, I = self.index.search(vectorized_input, top)
        return I[0], 1 - D[0]

    def save_index(self, file_path: str) -> None:
        """Save the FAISS index to disk.

        Args:
            file_path: The path where the index will be saved.
        """
        if hasattr(self, "index"):
            faiss.write_index(self.index, file_path)
        else:
            raise AttributeError(
                "The index has not been built yet. Build the index using `build_index` method first."
            )

    def load_index(self, file_path: str) -> None:
        """Load a previously saved FAISS index from disk.

        Args:
            file_path: The path where the index is stored.
        """
        if os.path.exists(file_path):
            self.index = faiss.read_index(file_path)
        else:
            raise FileNotFoundError(f"The specified file '{file_path}' does not exist.")

    @staticmethod
    def measure_time(func: Callable, *args, **kwargs) -> Tuple[float, Any]:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        return elapsed_time, result

    @staticmethod
    def measure_memory_usage() -> float:
        process = psutil.Process(os.getpid())
        ram = process.memory_info().rss
        return ram / (1024**2)

    def timed_train(self, data: List[str]) -> Tuple[float, float]:
        start_time = time.time()
        embeddings = self.encode(data)
        self.build_index(embeddings)
        end_time = time.time()
        elapsed_time = end_time - start_time
        memory_usage = self.measure_memory_usage()
        logging.info(
            "Training time: %.2f seconds on device: %s", elapsed_time, self.device
        )
        logging.info("Training memory usage: %.2f MB", memory_usage)
        return elapsed_time, memory_usage

    def timed_infer(self, query: str, top: int) -> Tuple[float, float]:
        start_time = time.time()
        _, _ = self.search(query, top)
        end_time = time.time()
        elapsed_time = end_time - start_time
        memory_usage = self.measure_memory_usage()
        logging.info(
            "Inference time: %.2f seconds on device: %s", elapsed_time, self.device
        )
        logging.info("Inference memory usage: %.2f MB", memory_usage)
        return elapsed_time, memory_usage

    def timed_load_index(self, file_path: str) -> float:
        start_time = time.time()
        self.load_index(file_path)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(
            "Index loading time: %.2f seconds on device: %s", elapsed_time, self.device
        )
        return elapsed_time
