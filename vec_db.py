from typing import Annotated
from sklearn.cluster import KMeans
import heapq
import struct
import numpy as np
import os

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70

class VecDB:
    def __init__(self, database_file_path="saved_db.dat", index_file_path="index.dat", new_db=True, db_size=None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path

        # Calculate the number of records in the database
        record_size = struct.calcsize(f"I{DIMENSION}f")
        file_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        n_records = file_size // record_size

        # Dynamically determine the number of probes and clusters
        if n_records <= 10 ** 6:
            self.n_probe = 5
            self.n_clusters_level_1 = 150
            self.n_clusters_level_2 = 10
        elif n_records <= 10 * 10 ** 6:
            self.n_probe = 8
            self.n_clusters_level_1 = 5000
            self.n_clusters_level_2 = 100
        elif n_records <= 15 * 10 ** 6:
            self.n_probe = 256
            self.n_clusters_level_1 = 7000
            self.n_clusters_level_2 = 500
        elif n_records <= 20 * 10 ** 6:
            self.n_probe = 64
            self.n_clusters_level_1 = 22000
            self.n_clusters_level_2 = 1000
        else:
            self.n_probe = 3
            self.n_clusters_level_1 = 20
            self.n_clusters_level_2 = 5

        self.centroids_level_1 = None
        self.centroids_level_2 = {}

        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)
        else:
            self.load_centroids()

    def load_centroids(self):
        centroids_path_level_1 = os.path.join(self.index_path, "centroids_level_1.bin")
        self.centroids_level_1 = self._load_centroids_from_file(centroids_path_level_1)

        for cluster_id in range(self.n_clusters_level_1):
            centroids_path_level_2 = os.path.join(self.index_path, f"centroids_level_2_cluster{cluster_id}.bin")
            if os.path.exists(centroids_path_level_2):
                self.centroids_level_2[cluster_id] = self._load_centroids_from_file(centroids_path_level_2)

    def _save_centroids_to_file(self, centroids, filename):
        """
        Save centroid data to a binary file.

        Args:
            centroids (np.ndarray): The array of centroid vectors.
            filename (str): The name of the file to save the centroids.
        """
        filepath = os.path.join(self.index_path, filename)
        with open(filepath, "wb") as f:
            for centroid in centroids:
                f.write(struct.pack(f"{len(centroid)}f", *centroid))

    def _load_centroids_from_file(self, filepath):
        """
        Load centroid data from a binary file.

        Args:
            filepath (str): The path to the centroid file.

        Returns:
            np.ndarray: Array of loaded centroids.
        """
        centroids = []
        with open(filepath, "rb") as f:
            while True:
                data = f.read(DIMENSION * struct.calcsize("f"))
                if not data:
                    break
                centroid = struct.unpack(f"{DIMENSION}f", data)
                centroids.append(centroid)
        return np.array(centroids)

    def _IVF_index(self, data, index_dir):
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)

        # First-level clustering
        print("Running first-level K-Means clustering...")
        kmeans_level_1 = KMeans(n_clusters=self.n_clusters_level_1, random_state=42, n_init='auto')
        vector_data = np.array([vector[1] for vector in data])
        labels_level_1 = kmeans_level_1.fit_predict(vector_data)
        self.centroids_level_1 = kmeans_level_1.cluster_centers_

        self._save_centroids_to_file(self.centroids_level_1, "centroids_level_1.bin")

        # Second-level clustering
        cluster_data_level_2 = {i: [] for i in range(self.n_clusters_level_1)}
        for i, label in enumerate(labels_level_1):
            cluster_data_level_2[label].append(data[i])

        for cluster_id, vectors in cluster_data_level_2.items():
            if len(vectors) == 0:
                continue

            print(f"Running second-level K-Means clustering for cluster {cluster_id}...")
            kmeans_level_2 = KMeans(n_clusters=min(self.n_clusters_level_2, len(vectors)), random_state=42, n_init='auto')
            vectors_only = np.array([v[1] for v in vectors])
            labels_level_2 = kmeans_level_2.fit_predict(vectors_only)
            self.centroids_level_2[cluster_id] = kmeans_level_2.cluster_centers_

            self._save_centroids_to_file(kmeans_level_2.cluster_centers_, f"centroids_level_2_cluster{cluster_id}.bin")

            # Save second-level clusters to disk
            sub_clusters = {i: [] for i in range(len(kmeans_level_2.cluster_centers_))}
            for i, sub_label in enumerate(labels_level_2):
                sub_clusters[sub_label].append(vectors[i][0])  # Only store IDs

            for sub_cluster_id, vector_ids in sub_clusters.items():
                sub_cluster_path = os.path.join(index_dir, f"cluster_{cluster_id}_{sub_cluster_id}.bin")
                self._write_cluster_file(sub_cluster_path, vector_ids)

    def _semantic_query_ivf(self, query_vector, top_k=5, index_dir=None):
        if self.centroids_level_1 is None:
            raise ValueError("Index must be built before querying.")

        # First-level search
        print("Finding nearest first-level clusters...")
        cluster_distances_level_1 = np.linalg.norm(self.centroids_level_1 - query_vector, axis=1)
        nearest_clusters_level_1 = np.argsort(cluster_distances_level_1)[:self.n_probe]

        top_scores_heap = []
        for cluster_id in nearest_clusters_level_1:
            if cluster_id not in self.centroids_level_2:
                continue

            # Second-level search
            print(f"Searching second-level clusters for cluster {cluster_id}...")
            cluster_distances_level_2 = np.linalg.norm(self.centroids_level_2[cluster_id] - query_vector, axis=1)
            nearest_clusters_level_2 = np.argsort(cluster_distances_level_2)[:self.n_probe]

            for sub_cluster_id in nearest_clusters_level_2:
                cluster_path = os.path.join(index_dir, f"cluster_{cluster_id}_{sub_cluster_id}.bin")
                if not os.path.exists(cluster_path):
                    continue

                with open(cluster_path, "rb") as f:
                    while True:
                        chunk = f.read(4)
                        if not chunk:
                            break

                        vector_id = struct.unpack('I', chunk)[0]
                        vector = self.get_one_row(vector_id)

                        similarity = self._cal_score(query_vector, vector)

                        if len(top_scores_heap) < top_k:
                            heapq.heappush(top_scores_heap, (similarity, vector_id))
                        else:
                            heapq.heappushpop(top_scores_heap, (similarity, vector_id))

        print("Selecting top-k candidates...")
        top_scores_heap.sort(reverse=True)
        top_candidates = [vector_id for _, vector_id in top_scores_heap]

        return top_candidates

    def retrieve(self, query, top_k=5):
        return self._semantic_query_ivf(query, top_k=top_k, index_dir=self.index_path)


    def _write_cluster_file(self, file_path, vector_ids):
        with open(file_path, "wb") as f:
            for vector_id in vector_ids:
                f.write(struct.pack('I', vector_id))  # Write ID as 4-byte unsigned int

    

    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
        vectors = [(idx, vector) for idx, vector in enumerate(vectors)]
        self._IVF_index(vectors, self.index_path)

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, 70)]):
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()

    def get_one_row(self, row_num: int) -> np.ndarray:
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            return f"An error occurred: {e}"

    def get_all_rows(self) -> np.ndarray:
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)


    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity
