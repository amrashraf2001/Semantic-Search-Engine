import struct
import numpy as np
import os
import faiss
from collections import defaultdict
import mmap

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70

class VecDB:
    def __init__(self, database_file_path="saved_db.dat", index_file_path="index_dir", new_db=True, db_size=None):
        self.db_path = database_file_path
        self.index_file_path = index_file_path
        self.DIMENSION = DIMENSION
        self.record_format = 'If' + 'f' * (DIMENSION - 1)
        self.record_size = struct.calcsize(self.record_format)
        
        file_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        n_records = file_size // self.record_size

        # Optimized clustering parameters
        if n_records <= 10 ** 6:
            self.nlist = 64
            self.n_probe = 8
        else:
            self.nlist = 128
            self.n_probe = 16

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
        
        # Initialize memory mapping
        self._init_mmap()
        
        if not new_db:
            self.load_index()

    def _init_mmap(self):
        try:
            if hasattr(self, 'mmap'):
                self.mmap.close()
            if hasattr(self, 'file'):
                self.file.close()
            
            self.file = open(self.db_path, 'rb')
            self.mmap = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception as e:
            print(f"Error initializing mmap: {e}")
            self.file = None
            self.mmap = None

    def __del__(self):
        if hasattr(self, 'mmap') and self.mmap is not None:
            self.mmap.close()
        if hasattr(self, 'file') and self.file is not None:
            self.file.close()

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

        # Read vectors in chunks for clustering
        chunk_size = min(total_records, 1_000_000)
        vectors = np.zeros((chunk_size, DIMENSION), dtype=np.float32)
        ids = np.zeros(chunk_size, dtype=np.int32)

        with open(self.db_path, "rb") as f:
            for i in range(chunk_size):
                data = f.read(self.record_size)
                unpacked = struct.unpack(self.record_format, data)
                ids[i] = unpacked[0]
                vectors[i] = unpacked[1:]

        # Perform clustering
        print("Performing clustering...")
        kmeans = faiss.Kmeans(DIMENSION, self.nlist, niter=20, verbose=True, seed=DB_SEED_NUMBER)
        kmeans.train(vectors)

        # Compute assignments in batches
        cluster_index = defaultdict(list)
        batch_size = 100_000
        
        for start in range(0, total_records, batch_size):
            end = min(start + batch_size, total_records)
            batch_vectors = np.zeros((end - start, DIMENSION), dtype=np.float32)
            batch_ids = np.zeros(end - start, dtype=np.int32)
            
            with open(self.db_path, "rb") as f:
                f.seek(start * self.record_size)
                for i in range(end - start):
                    data = f.read(self.record_size)
                    unpacked = struct.unpack(self.record_format, data)
                    batch_ids[i] = unpacked[0]
                    batch_vectors[i] = unpacked[1:]
            
            # Compute distances to centroids
            distances = np.zeros((len(batch_vectors), self.nlist), dtype=np.float32)
            for i in range(self.nlist):
                distances[:, i] = np.linalg.norm(batch_vectors - kmeans.centroids[i], axis=1)
            labels = np.argmin(distances, axis=1)
            
            # Update cluster index
            for idx, label in enumerate(labels):
                cluster_index[int(label)].append(int(batch_ids[idx]))

        # Save index
        np.save(os.path.join(self.index_file_path, "centroids.npy"), kmeans.centroids)
        np.save(os.path.join(self.index_file_path, "cluster_index.npy"), dict(cluster_index))
        print(f"Index saved to {self.index_file_path}")

    def load_index(self):
        centroids_path = os.path.join(self.index_file_path, "centroids.npy")
        cluster_index_path = os.path.join(self.index_file_path, "cluster_index.npy")

        if not all(os.path.exists(p) for p in [centroids_path, cluster_index_path]):
            raise FileNotFoundError("Index files not found. Please build the index first.")

        self.centroids = np.load(centroids_path)
        self.cluster_index = defaultdict(list, np.load(cluster_index_path, allow_pickle=True).item())
        print(f"Index loaded from {self.index_file_path}")

    def get_vector_mmap(self, row_num: int) -> np.ndarray:
        if self.mmap is None:
            self._init_mmap()
            if self.mmap is None:
                return self.get_one_row(row_num)
                
        offset = row_num * self.record_size
        self.mmap.seek(offset)
        data = self.mmap.read(self.record_size)
        unpacked = struct.unpack(self.record_format, data)
        return np.array(unpacked[1:], dtype=np.float32)

    def retrieve(self, query_vector, top_k=5):
        if not hasattr(self, 'centroids'):
            self.load_index()

        query_vector = np.array(query_vector, dtype=np.float32).reshape(-1)
    
        # Find nearest clusters using numpy operations
        distances = np.linalg.norm(self.centroids - query_vector, axis=1)
        nearest_clusters = np.argpartition(distances, self.n_probe)[:self.n_probe]

        # Get candidate IDs
        candidate_ids = []
        for cluster_id in nearest_clusters:
            candidate_ids.extend(self.cluster_index[cluster_id])

        if not candidate_ids:
            return []

        # Batch process candidates
        batch_size = 1000
        best_distances = None  # Changed from list to None
        best_ids = []
    
        for i in range(0, len(candidate_ids), batch_size):
            batch_ids = candidate_ids[i:i + batch_size]
            batch_vectors = np.zeros((len(batch_ids), DIMENSION), dtype=np.float32)
        
            # Load vectors using memory mapping
            for j, idx in enumerate(batch_ids):
                batch_vectors[j] = self.get_vector_mmap(idx)
        
            # Compute distances
            batch_distances = np.linalg.norm(batch_vectors - query_vector, axis=1)
        
            # Keep track of top-k
            if best_distances is None:  # Changed condition
                best_indices = np.argsort(batch_distances)[:top_k]
                best_distances = batch_distances[best_indices]
                best_ids = [batch_ids[i] for i in best_indices]
            else:
                # Merge with existing results
                all_distances = np.concatenate([best_distances, batch_distances])
                all_ids = best_ids + batch_ids
                top_indices = np.argpartition(all_distances, top_k)[:top_k]
                best_distances = all_distances[top_indices]
                best_ids = [all_ids[i] for i in top_indices]

        return best_ids

    def insert_records(self, rows: np.ndarray):
        # Close existing mmap before writing
        if hasattr(self, 'mmap') and self.mmap is not None:
            self.mmap.close()
        if hasattr(self, 'file') and self.file is not None:
            self.file.close()

        num_old_records = self._get_num_records()
        
        # Write to file
        with open(self.db_path, "ab") as f:
            for i, vector in enumerate(rows, start=num_old_records):
                record = struct.pack(self.record_format, i, *vector)
                f.write(record)

        # Compute cluster assignments
        distances = np.zeros((len(rows), len(self.centroids)), dtype=np.float32)
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.linalg.norm(rows - centroid, axis=1)
        labels = np.argmin(distances, axis=1)
        
        # Update index
        for i, label in enumerate(labels):
            vec_id = num_old_records + i
            self.cluster_index[label].append(vec_id)

        # Save updated index
        np.save(os.path.join(self.index_file_path, "cluster_index.npy"), dict(self.cluster_index))
        
        # Reinitialize mmap
        self._init_mmap()

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