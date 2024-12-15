from typing import Dict, List, Annotated
from sklearn.cluster import KMeans
from heapq import nlargest
import heapq
import struct
import numpy as np
import os

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70

# class ProductQuantizer:
#     def __init__(self, n_subvectors=2, n_clusters=256):
#         self.n_subvectors = n_subvectors
#         self.n_clusters = n_clusters
#         self.kmeans_list = [KMeans(n_clusters=self.n_clusters, random_state=42) for _ in range(n_subvectors)]

#     def fit(self, data):
#         subvector_length = data.shape[1] // self.n_subvectors
#         self.centroids = []

#         for i, kmeans in enumerate(self.kmeans_list):
#             subdata = data[:, i * subvector_length:(i + 1) * subvector_length]
#             kmeans.fit(subdata)
#             self.centroids.append(kmeans.cluster_centers_)

#     def encode(self, vector):
#         subvector_length = vector.shape[0] // self.n_subvectors
#         codes = []

#         for i, kmeans in enumerate(self.kmeans_list):
#             subvector = vector[i * subvector_length:(i + 1) * subvector_length]
#             code = kmeans.predict([subvector])
#             codes.append(code[0])

#         return codes

#     def decode(self, codes):
#         subvector_length = self.centroids[0].shape[1]
#         vector = np.zeros(subvector_length * self.n_subvectors)

#         for i, code in enumerate(codes):
#             centroid = self.centroids[i][code]
#             vector[i * subvector_length:(i + 1) * subvector_length] = centroid

#         return vector

class VecDB:
    def __init__(self, database_file_path="saved_db.dat", index_file_path="index.dat", new_db=True, db_size=None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path

        record_size = struct.calcsize(f"I{70}f")
        file_size = os.path.getsize(self.db_path)
        n_records = file_size // record_size

        if n_records <= 10 ** 6:
            n_probes = 5
            self.number_of_clusters = 10
        elif n_records <= 10 * 10 ** 6:
            n_probes = 30
            self.number_of_clusters = 200
        elif n_records <= 15 * 10 ** 6:
            n_probes = 256
            self.number_of_clusters = 500
        elif n_records <= 20 * 10 ** 6:
            n_probes = 64
            self.number_of_clusters = 8000
        else:
            n_probes = 3
            self.number_of_clusters = 10

        self.n_probe = n_probes

        self.n_clusters = self.number_of_clusters
        self.centroids = None
        self.index_files = []
        # self.pq = ProductQuantizer(n_subvectors=4, n_clusters=256)

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
            # Load centroids from a file
            centroids_path = os.path.join(self.index_path, "centroids.bin")
            self.centroids = []
            predefined_length = DIMENSION  # Assuming the predefined length is the same as DIMENSION

            with open(centroids_path, "rb") as f:
                while True:
                    # Define the size of each centroid (adjust based on the number of floats per centroid)
                    centroid_size = len(self.centroids[0]) if self.centroids else predefined_length
                    bytes_to_read = centroid_size * struct.calcsize("f")
                    data = f.read(bytes_to_read)
                    if not data:
                        break
                    centroid = struct.unpack(f"{centroid_size}f", data)
                    self.centroids.append(centroid)

    def _IVF_index(self, data, index_dir):
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)

        print("Running K-Means clustering...")
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        vector_data = [vector[1] for vector in data]
        labels = kmeans.fit_predict(vector_data)
        self.centroids = kmeans.cluster_centers_

        # Save centroids to a file
        centroids_path = os.path.join(index_dir, "centroids.bin")
        with open(centroids_path, "wb") as f:
            for centroid in self.centroids:
                f.write(struct.pack(f"{len(centroid)}f", *centroid))

        # Group vector IDs by cluster and save them
        cluster_files = [os.path.join(index_dir, f"cluster{i}.bin") for i in range(self.n_clusters)]
        cluster_data = {i: [] for i in range(self.n_clusters)}

        for i, label in enumerate(labels):
            cluster_data[label].append(data[i][0])  # Append only the vector ID

        for cluster_id, vector_ids in cluster_data.items():
            with open(cluster_files[cluster_id], "wb") as f:
                for vector_id in vector_ids:
                    f.write(struct.pack('I', vector_id))  # Write ID as 4-byte unsigned int

        self.index_files = cluster_files
        print("Indexing complete.")



    def _semantic_query_ivf(self, query_vector, top_k=5, index_dir=None):
        query_vector = query_vector.squeeze()
        if index_dir is None or self.centroids is None:
            raise ValueError("Index must be built before querying.")

        print("Calculating nearest clusters...")
        cluster_distances = np.linalg.norm(self.centroids - query_vector, axis=1)
        nearest_clusters = np.argsort(cluster_distances)[:self.n_probe]

        # Use a heap to maintain top-k scores efficiently
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

                    # Compute similarity
                    similarity = self._cal_score(query_vector, vector)

                    # Maintain top-k scores using a heap
                    if len(top_scores_heap) < top_k:
                        heapq.heappush(top_scores_heap, (similarity, vector_id))
                    else:
                        heapq.heappushpop(top_scores_heap, (similarity, vector_id))

        print("Selecting top-k candidates...")
        # Extract and sort the final top-k candidates
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
        # TODO: might change to call insert in the index, if you need

    def get_one_row(self, row_num: int) -> np.ndarray:
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            return f"An error occurred: {e}"


    def get_all_rows(self) -> np.ndarray:
        # Take care this load all the data in memory
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)
    
    def retrieve(self, query, top_k = 5):
    
        results = np.array(self._semantic_query_ivf(query, top_k=top_k, index_dir=self.index_path))  
        return results
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    # def _build_index(self):
    #     # Placeholder for index building logic
    #     pass


