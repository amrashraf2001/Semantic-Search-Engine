import struct
import numpy as np
import os
import faiss
from collections import defaultdict

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70

def read_binary_file_chunk(file_path, start_index, chunk_size):
    record_format = 'If' + 'f' * (DIMENSION - 1)
    record_size = struct.calcsize(record_format)
    data = []
    
    with open(file_path, "rb") as f:
        f.seek(start_index * record_size)
        buffer = f.read(record_size * chunk_size)
        for i in range(0, len(buffer), record_size):
            chunk = buffer[i:i + record_size]
            if len(chunk) != record_size:
                break
            unpacked = struct.unpack(record_format, chunk)
            vec_id = int(unpacked[0])
            vector = np.array(unpacked[1:], dtype=np.float32)
            data.append((vec_id, vector))
    return data

def perform_kmeans(vectors, n_clusters):
    kmeans = faiss.Kmeans(DIMENSION, n_clusters, niter=20, verbose=True, seed=DB_SEED_NUMBER)
    kmeans.train(vectors)
    
    # Compute cluster assignments
    distances = np.zeros((len(vectors), n_clusters), dtype=np.float32)
    for i in range(n_clusters):
        distances[:, i] = np.linalg.norm(vectors - kmeans.centroids[i], axis=1)
    labels = np.argmin(distances, axis=1)
    
    return kmeans.centroids, labels

class VecDB:
    def __init__(self, database_file_path="saved_db.dat", index_file_path="index_dir", new_db=True, db_size=None):
        self.db_path = database_file_path
        self.index_file_path = index_file_path
        self.DIMENSION = DIMENSION
        self.record_format = 'If' + 'f' * (DIMENSION - 1)
        self.record_size = struct.calcsize(self.record_format)

        file_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        n_records = file_size // self.record_size

        # Optimized parameters for better speed
        if n_records <= 10 ** 6:
            self.nlist = 100
            self.n_probe = 10
        elif n_records <= 10 * 10 ** 6:
            self.nlist = 256
            self.n_probe = 16
        else:
            self.nlist = 512
            self.n_probe = 32

        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            if os.path.exists(self.index_file_path):
                for f in os.listdir(self.index_file_path):
                    os.remove(os.path.join(self.index_file_path, f))
            else:
                os.makedirs(self.index_file_path)
            self.generate_database(db_size)
            self.build_index()
        else:
            self.load_index()

    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        with open(self.db_path, "wb") as f:
            for i, vector in enumerate(vectors):
                record = struct.pack(self.record_format, i, *vector)
                f.write(record)

    def build_index(self):
        total_records = self._get_num_records()
        if total_records == 0:
            raise ValueError("No records found in database")

        sample_size = min(total_records, 5_000_000)
        sample_data = read_binary_file_chunk(self.db_path, 0, sample_size)
        vector_data = np.array([v[1] for v in sample_data], dtype=np.float32)
        
        print("Performing clustering...")
        centroids, labels = perform_kmeans(vector_data, self.nlist)
        
        # Build inverted index
        self.cluster_index = defaultdict(list)
        self.vector_data = {}
        
        for idx, (vec_id, vector) in enumerate(sample_data):
            cluster_id = labels[idx]
            self.cluster_index[cluster_id].append(vec_id)
            self.vector_data[vec_id] = vector

        # Save index
        np.save(os.path.join(self.index_file_path, "centroids.npy"), centroids)
        np.save(os.path.join(self.index_file_path, "cluster_index.npy"), dict(self.cluster_index))
        np.save(os.path.join(self.index_file_path, "vector_data.npy"), self.vector_data)
        print(f"Index saved to {self.index_file_path}")

    def load_index(self):
        centroids_path = os.path.join(self.index_file_path, "centroids.npy")
        cluster_index_path = os.path.join(self.index_file_path, "cluster_index.npy")
        vector_data_path = os.path.join(self.index_file_path, "vector_data.npy")

        if not all(os.path.exists(p) for p in [centroids_path, cluster_index_path, vector_data_path]):
            raise FileNotFoundError("Index files not found. Please build the index first.")

        self.centroids = np.load(centroids_path)
        self.cluster_index = defaultdict(list, np.load(cluster_index_path, allow_pickle=True).item())
        self.vector_data = np.load(vector_data_path, allow_pickle=True).item()
        print(f"Index loaded from {self.index_file_path}")

    def retrieve(self, query_vector, top_k=5):
        if not hasattr(self, 'centroids'):
            self.load_index()

        query_vector = np.array(query_vector, dtype=np.float32)
        
        # Find nearest clusters
        centroid_distances = np.linalg.norm(self.centroids - query_vector, axis=1)
        nearest_clusters = np.argpartition(centroid_distances, self.n_probe)[:self.n_probe]

        # Collect candidates
        candidate_ids = []
        for cluster_id in nearest_clusters:
            candidate_ids.extend(self.cluster_index[cluster_id])

        if not candidate_ids:
            return []

        # Compute distances efficiently
        candidate_vectors = np.array([self.vector_data[idx] for idx in candidate_ids], dtype=np.float32)
        distances = np.linalg.norm(candidate_vectors - query_vector, axis=1)
        
        # Get top-k efficiently
        if len(distances) <= top_k:
            indices = np.argsort(distances)
        else:
            indices = np.argpartition(distances, top_k)[:top_k]
            indices = indices[np.argsort(distances[indices])]

        return [candidate_ids[i] for i in indices]

    def insert_records(self, rows: np.ndarray):
        num_old_records = self._get_num_records()
        
        # Write to file
        with open(self.db_path, "ab") as f:
            for i, vector in enumerate(rows, start=num_old_records):
                record = struct.pack(self.record_format, i, *vector)
                f.write(record)

        # Assign to clusters
        distances = np.zeros((len(rows), len(self.centroids)), dtype=np.float32)
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.linalg.norm(rows - centroid, axis=1)
        
        labels = np.argmin(distances, axis=1)
        
        # Update index
        for i, (label, vector) in enumerate(zip(labels, rows)):
            vec_id = num_old_records + i
            self.cluster_index[label].append(vec_id)
            self.vector_data[vec_id] = vector

        # Save updated index
        np.save(os.path.join(self.index_file_path, "cluster_index.npy"), dict(self.cluster_index))
        np.save(os.path.join(self.index_file_path, "vector_data.npy"), self.vector_data)

    def get_one_row(self, row_num: int) -> np.ndarray:
        max_records = self._get_num_records()
        if row_num < 0 or row_num >= max_records:
            raise IndexError(f"Row number {row_num} is out of range.")
            
        with open(self.db_path, "rb") as f:
            f.seek(row_num * self.record_size)
            data = f.read(self.record_size)
            if len(data) != self.record_size:
                raise IndexError(f"Could not read vector at row {row_num}.")
            unpacked = struct.unpack(self.record_format, data)
            return np.array(unpacked[1:], dtype=np.float32)

    def get_all_rows(self) -> np.ndarray:
        num_records = self._get_num_records()
        vectors = np.empty((num_records, self.DIMENSION), dtype=np.float32)
        
        with open(self.db_path, "rb") as f:
            for i in range(num_records):
                data = f.read(self.record_size)
                if not data or len(data) != self.record_size:
                    raise IndexError(f"Could not read vector at row {i}.")
                unpacked = struct.unpack(self.record_format, data)
                vectors[i] = unpacked[1:]
        return vectors

    def _get_num_records(self) -> int:
        if not os.path.exists(self.db_path):
            return 0
        return os.path.getsize(self.db_path) // self.record_size