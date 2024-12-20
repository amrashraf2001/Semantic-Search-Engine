import struct
import numpy as np
import os
import faiss
import gc
import heapq

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize  
DIMENSION = 70  

class VecDB:
    def __init__(self, database_file_path="saved_db.dat", index_file_path="index_dir", new_db=True, db_size=None):
        self.db_path = database_file_path
        self.index_file_path = index_file_path
        self.DIMENSION = DIMENSION

        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            self.n_records = db_size
        else:
            file_size = os.path.getsize(self.db_path)
            self.n_records = file_size // (DIMENSION * ELEMENT_SIZE)

        self._set_clustering_parameters()

        if new_db:
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            if os.path.exists(self.index_file_path):
                for f in os.listdir(self.index_file_path):
                    os.remove(os.path.join(self.index_file_path, f))
            else:
                os.makedirs(self.index_file_path)
            self.generate_database(db_size)
        else:
            self.load_index()

    def _set_clustering_parameters(self):
        if self.n_records <= 1_000_000:
            self.nlist = 32
            self.n_probe = 4
        elif self.n_records <= 10_000_000:
            self.nlist = 32
            self.n_probe = 1
        elif self.n_records <= 15_000_000:
            self.nlist = 32
            self.n_probe = 1
        else:
            self.nlist = 32
            self.n_probe = 1

    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
        self.build_index()

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        with open(self.db_path, 'wb') as f:
            vectors.tofile(f)

    def build_index(self):
        print("Building index...")
        sample_size = min(self.n_records, 1_000_000)
        batch_size = 1_000

        # Read a sample of the vectors for clustering
        sample_vectors = self._read_vectors_from_file(0, sample_size)
        # Normalize vectors
        vector_norms = np.linalg.norm(sample_vectors, axis=1, keepdims=True)
        normalized_vectors = sample_vectors / vector_norms

        kmeans = faiss.Kmeans(self.DIMENSION, self.nlist, niter=20, verbose=True, seed=DB_SEED_NUMBER)
        kmeans.train(normalized_vectors.astype(np.float32))

        # Initialize cluster files
        cluster_files = {}
        for cluster_id in range(self.nlist):
            cluster_file_path = os.path.join(self.index_file_path, f"cluster_{cluster_id}.indices")
            cluster_files[cluster_id] = open(cluster_file_path, 'wb')

        # Assign vectors to clusters in batches
        for start in range(0, self.n_records, batch_size):
            end = min(start + batch_size, self.n_records)
            batch_vectors = self._read_vectors_from_file(start, end - start)
            batch_norms = np.linalg.norm(batch_vectors, axis=1, keepdims=True)
            normalized_batch = batch_vectors / batch_norms
            similarities = normalized_batch.dot(kmeans.centroids.T)
            labels = np.argmax(similarities, axis=1).astype('int32')

            for cluster_id in range(self.nlist):
                cluster_indices = np.where(labels == cluster_id)[0] + start
                cluster_indices.astype('uint32').tofile(cluster_files[cluster_id])

            # Free memory to keep RAM usage low
            del batch_vectors, batch_norms, normalized_batch, similarities, labels

        for f in cluster_files.values():
            f.close()

        np.save(os.path.join(self.index_file_path, "centroids.npy"), kmeans.centroids)
        print(f"Index saved to {self.index_file_path}")

    def _read_vectors_from_file(self, start_index: int, count: int) -> np.ndarray:
        # Read a batch of vectors from the file
        vectors = np.empty((count, self.DIMENSION), dtype=np.float32)
        offset = start_index * self.DIMENSION * ELEMENT_SIZE
        with open(self.db_path, 'rb') as f:
            f.seek(offset)
            bytes_to_read = count * self.DIMENSION * ELEMENT_SIZE
            data = f.read(bytes_to_read)
            vectors = np.frombuffer(data, dtype='float32').reshape(-1, self.DIMENSION)
        return vectors

    def load_index(self):
        centroids_path = os.path.join(self.index_file_path, "centroids.npy")

        if not os.path.exists(centroids_path):
            raise FileNotFoundError("Index files not found. Please build the index first.")

        self.centroids = np.load(centroids_path)
        print(f"Index loaded from {self.index_file_path}")

    def retrieve(self, query_vector, top_k=5):
        if not hasattr(self, 'centroids'):
            self.load_index()

        query_vector = query_vector.reshape(-1)
        query_norm = np.linalg.norm(query_vector)

        # Compute similarities to centroids
        centroid_norms = np.linalg.norm(self.centroids, axis=1)
        centroid_similarities = self.centroids.dot(query_vector) / (centroid_norms * query_norm)
        nearest_clusters = np.argpartition(centroid_similarities, -self.n_probe)[-self.n_probe:]

        max_RAM_usage = 20 * 1024 * 1024  # 20 MB in bytes
        vector_size = self.DIMENSION * ELEMENT_SIZE  # Size of one vector in bytes
        max_vectors_in_RAM = max_RAM_usage // vector_size  # Max number of vectors to load into RAM

        # To ensure we don't exceed RAM limit, we calculate chunk size
        chunk_size = max(1, int(max_vectors_in_RAM / self.n_probe / 2))  # Divided by n_probe and 2 for safety

        min_heap = []  # Min-heap to store top candidates

        for cluster_id in nearest_clusters:
            cluster_file_path = os.path.join(self.index_file_path, f"cluster_{cluster_id}.indices")

            if not os.path.exists(cluster_file_path):
                continue

            with open(cluster_file_path, 'rb') as f:
                dtype = np.dtype('uint32')
                itemsize = dtype.itemsize
                f.seek(0, os.SEEK_END)
                file_size = f.tell()
                num_items = file_size // itemsize
                f.seek(0)

                # Read candidate IDs in chunks to keep memory usage low
                for start in range(0, num_items, chunk_size):
                    count = min(chunk_size, num_items - start)
                    candidate_ids_chunk = np.fromfile(f, dtype=dtype, count=count)

                    # Read candidate vectors in batch
                    candidate_vectors = self._read_vectors_by_ids(candidate_ids_chunk)
                    candidate_norms = np.linalg.norm(candidate_vectors, axis=1)
                    similarities = candidate_vectors.dot(query_vector) / (candidate_norms * query_norm)

                    # Update the min-heap with candidate similarities
                    for sim, candidate_id in zip(similarities, candidate_ids_chunk):
                        if len(min_heap) < top_k:
                            heapq.heappush(min_heap, (sim, candidate_id))
                        else:
                            if sim > min_heap[0][0]:
                                heapq.heappushpop(min_heap, (sim, candidate_id))

                    # Free memory used by batch variables
                    del candidate_ids_chunk, candidate_vectors, candidate_norms, similarities

        if not min_heap:
            # If no candidates found, return default indices
            return list(range(min(top_k, self.n_records)))

        # Retrieve top candidates from the heap
        top_candidates = heapq.nlargest(top_k, min_heap)
        result = [candidate_id for similarity, candidate_id in top_candidates]

        return result

    def _read_vectors_by_ids(self, ids: np.ndarray) -> np.ndarray:
        count = len(ids)
        vectors = np.empty((count, self.DIMENSION), dtype=np.float32)
        with open(self.db_path, 'rb') as f:
            for i, vector_id in enumerate(ids):
                offset = vector_id * self.DIMENSION * ELEMENT_SIZE
                f.seek(offset)
                vector_bytes = f.read(self.DIMENSION * ELEMENT_SIZE)
                vectors[i, :] = np.frombuffer(vector_bytes, dtype='float32', count=self.DIMENSION)
        return vectors

    def get_one_row(self, row_num: int) -> np.ndarray:
        if row_num < 0 or row_num >= self.n_records:
            raise IndexError(f"Row number {row_num} is out of range.")
        return self._read_vector(row_num)

    def _read_vector(self, vector_id: int) -> np.ndarray:
        offset = vector_id * self.DIMENSION * ELEMENT_SIZE
        with open(self.db_path, 'rb') as f:
            f.seek(offset)
            vector_bytes = f.read(self.DIMENSION * ELEMENT_SIZE)
            vector = np.frombuffer(vector_bytes, dtype='float32', count=self.DIMENSION)
        return vector

    def get_all_rows(self) -> np.ndarray:
        vectors = np.empty((self.n_records, self.DIMENSION), dtype=np.float32)
        with open(self.db_path, 'rb') as f:
            data = f.read()
            vectors = np.frombuffer(data, dtype='float32').reshape(-1, self.DIMENSION)
        return vectors

    def _get_num_records(self) -> int:
        return self.n_records

    def __del__(self):
        gc.collect()