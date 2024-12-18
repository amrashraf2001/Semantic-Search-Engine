import struct
import numpy as np
import os
import faiss

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70

def read_binary_file_chunk(file_path, start_index, chunk_size):
    """
    Read a chunk of (id, vector) records from the main DB file.
    Returns a list of tuples (id, vector).
    """
    record_format = f"I{DIMENSION}f"
    record_size = struct.calcsize(record_format)
    data = []
    with open(file_path, "rb") as f:
        f.seek(start_index * record_size)
        for _ in range(chunk_size):
            record_bytes = f.read(record_size)
            if not record_bytes:
                break
            unpacked = struct.unpack(record_format, record_bytes)
            vec_id = unpacked[0]
            vector = np.array(unpacked[1:], dtype=np.float32)
            data.append((vec_id, vector))
    return data

def perform_kmeans(vectors, n_clusters):
    """
    Perform k-means clustering using faiss's KMeans implementation.
    Returns the centroids and the cluster assignments.
    """
    kmeans = faiss.Kmeans(DIMENSION, n_clusters, niter=20, verbose=True, seed=DB_SEED_NUMBER)
    kmeans.train(vectors)
    centroids = kmeans.centroids

    # Assign each vector to a cluster
    _, labels = kmeans.index.search(vectors, 1)  # Search returns distances and labels
    return centroids, labels.flatten()

class VecDB:
    def __init__(self, database_file_path="saved_db.dat", index_file_path="index_dir", new_db=True, db_size=None):
        self.db_path = database_file_path
        self.index_file_path = index_file_path
        self.DIMENSION = DIMENSION

        # Determine number of records
        record_size = (self.DIMENSION + 1) * ELEMENT_SIZE
        file_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        n_records = file_size // record_size

        # Dynamically determine the number of clusters
        if n_records <= 10 ** 6:
            self.nlist = 100
            self.n_probe = 10
        elif n_records <= 10 * 10 ** 6:
            self.nlist = 500
            self.n_probe = 50
        elif n_records <= 15 * 10 ** 6:
            self.nlist = 2000
            self.n_probe = 100
        elif n_records <= 20 * 10 ** 6:
            self.nlist = 5000
            self.n_probe = 200
        else:
            self.nlist = 10000
            self.n_probe = 500

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
        self.build_index()

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        """
        Write vectors to the binary database file.
        """
        record_format = f"I{self.DIMENSION}f"
        with open(self.db_path, "wb") as f:
            for i, v in enumerate(vectors):
                f.write(struct.pack(record_format, i, *v))

    def build_index(self):
        """
        Build the index by performing k-means clustering and saving centroids and inverted indices.
        """
        total_records = self._get_num_records()
        sample_size = min(total_records, 10_000_000)
        sample_data = read_binary_file_chunk(self.db_path, 0, sample_size)
        vector_data = np.array([v[1] for v in sample_data], dtype=np.float32)

        print("Performing k-means clustering...")
        centroids, labels = perform_kmeans(vector_data, self.nlist)

        # Build an inverted index with only IDs
        self.inverted_index = {i: [] for i in range(self.nlist)}
        for idx, cluster_id in enumerate(labels):
            self.inverted_index[cluster_id].append(sample_data[idx][0])

        # Save centroids and inverted index to disk
        np.save(os.path.join(self.index_file_path, "centroids.npy"), centroids)
        np.save(os.path.join(self.index_file_path, "inverted_index.npy"), self.inverted_index)
        print(f"Index saved to {self.index_file_path}")

    def load_index(self):
        """
        Load the index from disk.
        """
        centroids_path = os.path.join(self.index_file_path, "centroids.npy")
        inverted_index_path = os.path.join(self.index_file_path, "inverted_index.npy")

        if not os.path.exists(centroids_path) or not os.path.exists(inverted_index_path):
            raise FileNotFoundError("Index files not found. Please build the index first.")

        self.centroids = np.load(centroids_path)
        self.inverted_index = np.load(inverted_index_path, allow_pickle=True).item()
        print(f"Index loaded from {self.index_file_path}")

    def retrieve(self, query_vector, top_k=5):
        """
        Retrieve the top_k closest vectors to the query vector.
        """
        if not hasattr(self, 'centroids') or not hasattr(self, 'inverted_index'):
            self.load_index()

        query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)

        # Find the nearest centroid
        distances = np.linalg.norm(self.centroids - query_vector, axis=1)
        nearest_centroid = np.argmin(distances)

        # Retrieve IDs from the nearest centroid's cluster
        candidate_ids = self.inverted_index.get(nearest_centroid, [])

        # Load vectors for these IDs and compute exact distances
        candidates = [self.get_one_row(id_) for id_ in candidate_ids]
        candidates = np.array(candidates, dtype=np.float32)
        exact_distances = np.linalg.norm(candidates - query_vector, axis=1)

        # Get the top_k results
        top_k_indices = np.argsort(exact_distances)[:top_k]
        top_k_ids = [candidate_ids[i] for i in top_k_indices]
        return top_k_ids

    def insert_records(self, rows: np.ndarray):
        """
        Insert new vectors into the database and rebuild the index.
        """
        rows = np.array(rows, dtype=np.float32)

        num_old_records = self._get_num_records()
        record_format = f"I{self.DIMENSION}f"
        with open(self.db_path, "ab") as f:
            for i, v in enumerate(rows, start=num_old_records):
                f.write(struct.pack(record_format, i, *v))

        self.build_index()

    def get_one_row(self, row_num: int) -> np.ndarray:
        record_format = f"I{self.DIMENSION}f"
        record_size = struct.calcsize(record_format)
        max_records = self._get_num_records()
        if row_num < 0 or row_num >= max_records:
            raise IndexError(f"Row number {row_num} is out of range.")
        offset = row_num * record_size
        with open(self.db_path, "rb") as f:
            f.seek(offset)
            data = f.read(record_size)
            if len(data) != record_size:
                raise IndexError(f"Could not read vector at row {row_num}.")
            unpacked = struct.unpack(record_format, data)
            vector = np.array(unpacked[1:], dtype=np.float32)
            return vector

    def get_all_rows(self) -> np.ndarray:
        num_records = self._get_num_records()
        record_format = f"I{self.DIMENSION}f"
        record_size = struct.calcsize(record_format)
        vectors = np.empty((num_records, self.DIMENSION), dtype=np.float32)
        with open(self.db_path, "rb") as f:
            for i in range(num_records):
                data = f.read(record_size)
                if not data or len(data) != record_size:
                    raise IndexError(f"Could not read vector at row {i}.")
                unpacked = struct.unpack(record_format, data)
                vectors[i] = unpacked[1:]
        return vectors

    def _get_num_records(self) -> int:
        record_size = (self.DIMENSION + 1) * ELEMENT_SIZE
        return os.path.getsize(self.db_path) // record_size
