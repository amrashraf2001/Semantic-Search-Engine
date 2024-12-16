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

        record_size = struct.calcsize(f"I{70}f")
        file_size = os.path.getsize(self.db_path)
        n_records = file_size // record_size

        if n_records <= 10 ** 6:
            n_probes = 5
            self.number_of_clusters = 150  # Increased the cluster size for faster indexing
        elif n_records <= 10 * 10 ** 6:
            n_probes = 30
            self.number_of_clusters = 1000
        elif n_records <= 15 * 10 ** 6:
            n_probes = 256
            self.number_of_clusters = 2000
        elif n_records <= 20 * 10 ** 6:
            n_probes = 64
            self.number_of_clusters = 22000
        else:
            n_probes = 3
            self.number_of_clusters = 20

        self.n_probe = n_probes
        self.n_clusters = self.number_of_clusters
        self.centroids = None
        self.index_files = []

        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # delete the old DB file if exists
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)
        else:
            if not os.path.exists(self.index_path):
                vectors = self.get_all_rows()
                vectors = [(idx, vector) for idx, vector in enumerate(vectors)]
                self._IVF_index(vectors, self.index_path)
            centroids_path = os.path.join(self.index_path, "centroids.bin")
            self.centroids = []
            
            with open(centroids_path, "rb") as f:
                while True:
                    data = f.read(DIMENSION * struct.calcsize("f"))
                    if not data:
                        break
                    centroid = struct.unpack(f"{DIMENSION}f", data)
                    self.centroids.append(centroid)

    def _IVF_index(self, data, index_dir):
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)

        print("Running K-Means clustering...")
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init='auto')
        vector_data = np.array([vector[1] for vector in data])
        labels = kmeans.fit_predict(vector_data)
        self.centroids = kmeans.cluster_centers_

        # Save centroids to a file
        centroids_path = os.path.join(index_dir, "centroids.bin")
        with open(centroids_path, "wb") as f:
            for centroid in self.centroids:
                f.write(struct.pack(f"{len(centroid)}f", *centroid))

        # Create in-memory cluster storage
        cluster_data = {i: [] for i in range(self.n_clusters)}
        for i, label in enumerate(labels):
            cluster_data[label].append(data[i][0])

        # Save clusters to disk
        print("Saving clusters to disk...")
        for cluster_id, vector_ids in cluster_data.items():
            cluster_path = os.path.join(index_dir, f"cluster{cluster_id}.bin")
            self._write_cluster_file(cluster_path, vector_ids)

        self.index_files = [os.path.join(index_dir, f"cluster{i}.bin") for i in range(self.n_clusters)]
        print("Indexing complete.")

    def _write_cluster_file(self, file_path, vector_ids):
        with open(file_path, "wb") as f:
            for vector_id in vector_ids:
                f.write(struct.pack('I', vector_id))  # Write ID as 4-byte unsigned int

    def _semantic_query_ivf(self, query_vector, top_k=5, index_dir=None):
        query_vector = query_vector.squeeze()
        if index_dir is None or self.centroids is None:
            raise ValueError("Index must be built before querying.")

        print("Calculating nearest clusters...")
        cluster_distances = np.linalg.norm(self.centroids - query_vector, axis=1)
        nearest_clusters = np.argsort(cluster_distances)[:self.n_probe]

        top_scores_heap = []
        for cluster_id in nearest_clusters:
            cluster_path = os.path.join(index_dir, f"cluster{cluster_id}.bin")
            if not os.path.exists(cluster_path):
                continue

            with open(cluster_path, "rb") as f:
                while True:
                    chunk = f.read(4)  # Only read the ID (4 bytes)
                    if not chunk:
                        break

                    vector_id = struct.unpack('I', chunk)[0]  # Parse the vector ID
                    vector = self.get_one_row(vector_id)  # Fetch the vector from the database

                    similarity = self._cal_score(query_vector, vector)

                    if len(top_scores_heap) < top_k:
                        heapq.heappush(top_scores_heap, (similarity, vector_id))
                    else:
                        heapq.heappushpop(top_scores_heap, (similarity, vector_id))

        print("Selecting top-k candidates...")
        top_scores_heap.sort(reverse=True)  # Sort by similarity in descending order
        top_candidates = [vector_id for _, vector_id in top_scores_heap]

        return top_candidates

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

    def retrieve(self, query, top_k=5):
        results = np.array(self._semantic_query_ivf(query, top_k=top_k, index_dir=self.index_path))
        return results

    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity
