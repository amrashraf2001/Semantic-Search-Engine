%pip install faiss-cpu
import faiss
import struct
import numpy as np
import os

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

class VecDB:
    def __init__(self, database_file_path="saved_db.dat", index_file_path="index_dir", new_db=True, db_size=None):
        self.db_path = database_file_path
        self.index_file_path = index_file_path

        self.DIMENSION = DIMENSION

        # Determine number of records
        record_size = (self.DIMENSION + 1) * ELEMENT_SIZE
        file_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        n_records = file_size // record_size

        # Dynamically determine the number of probes and clusters
        # Adjust these parameters as needed
        if n_records <= 10 ** 6:
            self.nlist = 100  # Number of clusters
            self.n_probe = 10
            self.m = 14  # Number of subquantizers, adjusted to divide DIMENSION=70
        elif n_records <= 10 * 10 ** 6:
            self.nlist = 500
            self.n_probe = 50
            self.m = 35
        elif n_records <= 15 * 10 ** 6:
            self.nlist = 2000
            self.n_probe = 100
            self.m = 14
        elif n_records <= 20 * 10 ** 6:
            self.nlist = 5000
            self.n_probe = 200
            self.m = 14
        else:
            self.nlist = 10000
            self.n_probe = 500
            self.m = 14

        # Ensure DIMENSION is divisible by m
        if self.DIMENSION % self.m != 0:
            raise ValueError(f"DIMENSION {self.DIMENSION} must be divisible by number of subquantizers m={self.m}")

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
        vectors = rng.random((size, self.DIMENSION), dtype=np.float32)
        # Normalize vectors for cosine similarity
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
        self._write_vectors_to_file(vectors)

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        # Write ID + vector to the main DB file
        record_format = f"I{self.DIMENSION}f"
        with open(self.db_path, "wb") as f:
            for i, v in enumerate(vectors):
                f.write(struct.pack(record_format, i, *v))

    def build_index(self):
        # Read sample data for training
        total_records = self._get_num_records()
        sample_size = min(total_records, 100_000)
        sample_data = read_binary_file_chunk(self.db_path, 0, sample_size)
        vector_data = np.array([v[1] for v in sample_data], dtype=np.float32)

        # Build the index with IndexIVFPQ
        quantizer = faiss.IndexFlatIP(self.DIMENSION)  # Inner product for cosine similarity
        index = faiss.IndexIVFPQ(quantizer, self.DIMENSION, self.nlist, self.m, 8)  # 8 bits per code

        # Train the index
        print("Training IndexIVFPQ...")
        index.train(vector_data)

        # Add vectors to the index in chunks to save memory
        print("Adding vectors to the index...")
        chunk_size = 100_000
        for start in range(0, total_records, chunk_size):
            end = min(start + chunk_size, total_records)
            chunk_data = read_binary_file_chunk(self.db_path, start, end - start)
            vectors = np.array([v[1] for v in chunk_data], dtype=np.float32)
            ids = np.array([v[0] for v in chunk_data], dtype=np.int64)  # IDs must be int64
            index.add_with_ids(vectors, ids)

        # Save the index to disk
        index_path = os.path.join(self.index_file_path, "index_ivfpq.faiss")
        faiss.write_index(index, index_path)
        print(f"Index saved to {index_path}")

    def load_index(self):
        index_path = os.path.join(self.index_file_path, "index_ivfpq.faiss")
        if not os.path.exists(index_path):
            raise FileNotFoundError("Index file not found. Please build the index first.")
        self.index = faiss.read_index(index_path)
        print(f"Index loaded from {index_path}")
        self.index.nprobe = self.n_probe  # Set nprobe for search

    def retrieve(self, query_vector, top_k=5):
        if not hasattr(self, 'index'):
            self.load_index()

        # Normalize query vector for cosine similarity
        query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        query_vector /= np.linalg.norm(query_vector) + 1e-10

        # Perform the search
        scores, indices = self.index.search(query_vector, top_k)

        # Retrieve the top_k IDs
        ids = indices[0]
        return ids.tolist()

    def insert_records(self, rows: np.ndarray):
        # Normalize vectors
        rows = np.array(rows, dtype=np.float32)
        rows /= np.linalg.norm(rows, axis=1, keepdims=True) + 1e-10

        num_old_records = self._get_num_records()
        record_format = f"I{self.DIMENSION}f"
        with open(self.db_path, "ab") as f:
            for i, v in enumerate(rows, start=num_old_records):
                f.write(struct.pack(record_format, i, *v))

        # Add new vectors to the index
        self.load_index()  # Ensure index is loaded
        vectors = rows.astype(np.float32)
        ids = np.arange(num_old_records, num_old_records + len(rows), dtype=np.int64)
        self.index.add_with_ids(vectors, ids)
        # Optionally re-train or re-build the index if necessary

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