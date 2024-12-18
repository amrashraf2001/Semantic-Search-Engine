import struct
import numpy as np
import os
import faiss
import mmap

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70

class VecDB:
    def __init__(self, database_file_path="saved_db.dat", index_file_path="index_dir", new_db=True, db_size=None):
        self.db_path = database_file_path
        self.index_file_path = index_file_path
        self.DIMENSION = DIMENSION
        
        # Initialize size and clustering parameters first
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            self.n_records = db_size
        else:
            file_size = os.path.getsize(self.db_path)
            self.n_records = file_size // (DIMENSION * ELEMENT_SIZE)

        # Set clustering parameters
        self._set_clustering_parameters()

        # Handle database creation or loading
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
        if self.n_records <= 10 ** 6:
            self.nlist = 32
            self.n_probe = 4
        elif self.n_records <= 10 * 10 ** 6:
            self.nlist = 64
            self.n_probe = 8
        else:
            self.nlist = 128
            self.n_probe = 16

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
        
        # Read sample vectors efficiently
        vectors = self.data[:sample_size].copy()
        
        # Normalize vectors for clustering
        vector_norms = np.linalg.norm(vectors, axis=1)
        normalized_vectors = vectors / vector_norms[:, np.newaxis]
        
        # Perform clustering
        kmeans = faiss.Kmeans(self.DIMENSION, self.nlist, niter=20, verbose=True, seed=DB_SEED_NUMBER)
        kmeans.train(normalized_vectors.astype(np.float32))
        
        # Build labels array to store cluster assignments
        labels_path = os.path.join(self.index_file_path, "labels.memmap")
        labels_memmap = np.memmap(labels_path, dtype='int32', mode='w+', shape=(self.n_records,))
        
        batch_size = 100_000
        
        for start in range(0, self.n_records, batch_size):
            end = min(start + batch_size, self.n_records)
            batch_vectors = self.data[start:end]
            
            # Normalize batch vectors
            batch_norms = np.linalg.norm(batch_vectors, axis=1)
            normalized_batch = batch_vectors / batch_norms[:, np.newaxis]
            
            # Compute cosine similarities to centroids
            similarities = normalized_batch.dot(kmeans.centroids.T)
            labels = np.argmax(similarities, axis=1).astype('int32')
            
            # Store labels in memmap
            labels_memmap[start:end] = labels
        
        labels_memmap.flush()
        del labels_memmap  # Close the memmap

        # Save centroids
        np.save(os.path.join(self.index_file_path, "centroids.npy"), kmeans.centroids)
        print(f"Index saved to {self.index_file_path}")

    def load_index(self):
        centroids_path = os.path.join(self.index_file_path, "centroids.npy")
        labels_path = os.path.join(self.index_file_path, "labels.memmap")

        if not all(os.path.exists(p) for p in [centroids_path, labels_path]):
            raise FileNotFoundError("Index files not found. Please build the index first.")

        self.centroids = np.load(centroids_path)
        self.labels = np.memmap(labels_path, dtype='int32', mode='r', shape=(self.n_records,))
        self._init_data_access()
        print(f"Index loaded from {self.index_file_path}")

    def retrieve(self, query_vector, top_k=5):
        if not hasattr(self, 'centroids'):
            self.load_index()

        # Handle query vector shape and normalization
        query_vector = query_vector.reshape(-1)
        query_norm = np.linalg.norm(query_vector)
        
        # Find nearest clusters using cosine similarity
        centroid_norms = np.linalg.norm(self.centroids, axis=1)
        centroid_similarities = self.centroids.dot(query_vector) / (centroid_norms * query_norm)
        nearest_clusters = np.argpartition(centroid_similarities, -self.n_probe)[-self.n_probe:]

        # Collect candidate IDs by scanning labels in chunks
        candidate_ids = []
        labels_chunk_size = 10_000_000  # Adjust this based on available memory
        for cluster_id in nearest_clusters:
            cluster_candidate_ids = []
            for start in range(0, self.n_records, labels_chunk_size):
                end = min(start + labels_chunk_size, self.n_records)
                labels_chunk = self.labels[start:end]
                indices = np.where(labels_chunk == cluster_id)[0] + start
                cluster_candidate_ids.extend(indices.tolist())
                # Limit the number of candidates per cluster to avoid excessive RAM usage
                if len(cluster_candidate_ids) >= 100_000:  # Adjust this threshold as needed
                    break
            candidate_ids.extend(cluster_candidate_ids)
        
        # If not enough candidates, return default indices
        if len(candidate_ids) < top_k:
            return list(range(min(top_k, self.n_records)))

        # Process candidates in one batch
        candidate_ids = np.array(candidate_ids)
        candidate_vectors = self.data[candidate_ids]
        
        # Compute cosine similarities
        candidate_norms = np.linalg.norm(candidate_vectors, axis=1)
        similarities = candidate_vectors.dot(query_vector) / (candidate_norms * query_norm)
        
        # Get top_k results
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(-similarities[top_indices])]
        
        return candidate_ids[top_indices].tolist()

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