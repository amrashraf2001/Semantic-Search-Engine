import struct
import numpy as np
import os
import faiss
import gc
from typing import List
import time

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
            self.n_probe = 4
        elif self.n_records <= 15_000_000:  
            self.nlist = 32
            self.n_probe = 5  
        else: 
            self.nlist = 32
            self.n_probe = 5  

    def _init_data_access(self):
        if not hasattr(self, 'data'):
            self.data = np.memmap(self.db_path, dtype=np.float32, mode='r',
                                  shape=(self.n_records, self.DIMENSION))

    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
        self._init_data_access()
        self.build_index()

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()
        del mmap_vectors

    def build_index(self):
        print("Building index...")
        sample_size = min(self.n_records, 1_000_000)
        batch_size = 1_000

        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(self.n_records, self.DIMENSION))[:sample_size]
        vector_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized_vectors = vectors / vector_norms

        kmeans = faiss.Kmeans(self.DIMENSION, self.nlist, niter=20, verbose=True, seed=DB_SEED_NUMBER)
        kmeans.train(normalized_vectors.astype(np.float32))

        cluster_files = {}
        for cluster_id in range(self.nlist):
            cluster_file_path = os.path.join(self.index_file_path, f"cluster_{cluster_id}.indices")
            cluster_files[cluster_id] = open(cluster_file_path, 'wb')

        for start in range(0, self.n_records, batch_size):
            end = min(start + batch_size, self.n_records)
            batch_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(self.n_records, self.DIMENSION))[start:end]

            batch_norms = np.linalg.norm(batch_vectors, axis=1, keepdims=True)
            normalized_batch = batch_vectors / batch_norms

            similarities = normalized_batch.dot(kmeans.centroids.T)
            labels = np.argmax(similarities, axis=1).astype('int32')

            for cluster_id in range(self.nlist):
                cluster_indices = np.where(labels == cluster_id)[0] + start
                cluster_indices.astype('uint32').tofile(cluster_files[cluster_id])

        for f in cluster_files.values():
            f.close()

        np.save(os.path.join(self.index_file_path, "centroids.npy"), kmeans.centroids)
        print(f"Index saved to {self.index_file_path}")

    def load_index(self):
        centroids_path = os.path.join(self.index_file_path, "centroids.npy")

        if not os.path.exists(centroids_path):
            raise FileNotFoundError("Index files not found. Please build the index first.")

        self.centroids = np.load(centroids_path)
        self._init_data_access()
        print(f"Index loaded from {self.index_file_path}")

    def retrieve(self, query_vector, top_k=5):
        if not hasattr(self, 'centroids'):
            self.load_index()

        query_vector = query_vector.reshape(-1)
        query_norm = np.linalg.norm(query_vector)

        centroid_norms = np.linalg.norm(self.centroids, axis=1)
        centroid_similarities = self.centroids.dot(query_vector) / (centroid_norms * query_norm)
        nearest_clusters = np.argpartition(centroid_similarities, -self.n_probe)[-self.n_probe:]

        total_candidate_ids = []
        total_similarities = []
        max_candidates_per_cluster = 100_000  # Adjust as needed

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
                num_items_to_read = min(num_items, max_candidates_per_cluster)
                f.seek(0)

                # Define chunk_size
                chunk_size = 10000  # Adjust as needed
                per_cluster_candidate_ids = []
                per_cluster_similarities = []

                for start in range(0, num_items_to_read, chunk_size):
                    num_items_in_chunk = min(chunk_size, num_items_to_read - start)
                    candidate_ids_chunk = np.fromfile(f, dtype=dtype, count=num_items_in_chunk)

                    if len(candidate_ids_chunk) == 0:
                        continue

                    candidate_vectors = self.data[candidate_ids_chunk]

                    # Compute similarities
                    candidate_norms = np.linalg.norm(candidate_vectors, axis=1)
                    similarities = candidate_vectors.dot(query_vector) / (candidate_norms * query_norm)

                    # Keep track of per cluster top_k candidates
                    if top_k < len(similarities):
                        top_indices = np.argpartition(-similarities, top_k - 1)[:top_k]
                    else:
                        top_indices = np.argsort(-similarities)[:top_k]

                    per_cluster_candidate_ids.extend(candidate_ids_chunk[top_indices].tolist())
                    per_cluster_similarities.extend(similarities[top_indices].tolist())

                    # Delete variables to free RAM
                    del candidate_ids_chunk
                    del candidate_vectors
                    del similarities
                    del candidate_norms
                    del top_indices

                # After processing all chunks, keep top_k per cluster
                if len(per_cluster_similarities) > top_k:
                    per_cluster_similarities = np.array(per_cluster_similarities)
                    per_cluster_candidate_ids = np.array(per_cluster_candidate_ids)
                    top_indices = np.argpartition(-per_cluster_similarities, top_k - 1)[:top_k]
                    per_cluster_candidate_ids = per_cluster_candidate_ids[top_indices]
                    per_cluster_similarities = per_cluster_similarities[top_indices]
                else:
                    per_cluster_candidate_ids = np.array(per_cluster_candidate_ids)
                    per_cluster_similarities = np.array(per_cluster_similarities)

                # Accumulate total candidates
                total_candidate_ids.extend(per_cluster_candidate_ids.tolist())
                total_similarities.extend(per_cluster_similarities.tolist())

                # Delete variables
                del per_cluster_candidate_ids
                del per_cluster_similarities

        if len(total_candidate_ids) == 0:
            return list(range(min(top_k, self.n_records)))

        # Now select top k from all combined candidates
        total_similarities = np.array(total_similarities)
        total_candidate_ids = np.array(total_candidate_ids)
        if top_k < len(total_similarities):
            top_indices = np.argpartition(-total_similarities, top_k - 1)[:top_k]
        else:
            top_indices = np.argsort(-total_similarities)[:top_k]

        result = total_candidate_ids[top_indices].tolist()

        # Clean up
        del total_candidate_ids
        del total_similarities
        del top_indices

        return result

    def get_one_row(self, row_num: int) -> np.ndarray:
        if not hasattr(self, 'data'):
            self._init_data_access()
        if row_num < 0 or row_num >= self.n_records:
            raise IndexError(f"Row number {row_num} is out of range.")
        return self.data[row_num]

    def get_all_rows(self) -> np.ndarray:
        if not hasattr(self, 'data'):
            self._init_data_access()
        return self.data

    def _get_num_records(self) -> int:
        return self.n_records

    def __del__(self):
        if hasattr(self, 'data'):
            del self.data
        gc.collect()