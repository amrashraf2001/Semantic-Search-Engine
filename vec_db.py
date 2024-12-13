from typing import Dict, List, Annotated
from sklearn.cluster import KMeans
from heapq import nlargest
import struct
import numpy as np
import os

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70

class VecDB:
    def __init__(self, database_file_path = "saved_db.dat", index_file_path = "index.dat", new_db = True, db_size = None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        
        record_size=struct.calcsize(f"I{70}f")
        file_size = os.path.getsize(self.db_path)
        n_records = file_size // record_size
        if(n_records==10*10**3):
          n_probes=3
        elif(n_records==100*10**3):
          n_probes=10
        elif(n_records==10**6):
          n_probes=5
        elif(n_records==5*10**6):
          n_probes=15
        elif(n_records==10*10**6):
          n_probes=30
        elif(n_records==15*10**6):
          n_probes=256
        elif(n_records==20*10**6):
          n_probes=64
        else:
          n_probes=3
        self.n_probe = n_probes
        if(n_records==10*10**3):           
          self.number_of_clusters=10
        elif(n_records==100*10**3):
          self.number_of_clusters=50   
        elif(n_records==10**6):
          self.number_of_clusters=200
        elif(n_records==5*10**6):
          self.number_of_clusters=500
        elif(n_records==10*10**6):
          self.number_of_clusters=8000
        else:
          self.number_of_clusters=10
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
        
    
    def _IVF_index(self, data, index_dir):
        """
        Build the IVF index by clustering data and storing cluster information in binary files.
        
        Args:
            data (np.ndarray): Data vectors to index.
            index_dir (str): Directory to store index files.
        """
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

        # Group vectors by cluster and save them
        cluster_files = [os.path.join(index_dir, f"cluster{i}.bin") for i in range(self.n_clusters)]
        cluster_data = {i: [] for i in range(self.n_clusters)}

        for i, label in enumerate(labels):
            cluster_data[label].append(data[i])

        for cluster_id, vectors in cluster_data.items():
            # with open(cluster_files[cluster_id], "wb") as f:
            #     for vector in vectors:
            #         f.write(struct.pack(f"{len(vector)}f", *vector))
            with open(cluster_files[cluster_id], "wb") as f:
                for idx, vector in vectors:
                    # Pack the index (4 bytes unsigned int) and then the vector (70 floats)
                    f.write(struct.pack('I', idx))  # Write index as 4-byte unsigned int
                    f.write(struct.pack(f"{len(vector)}f", *vector))        
                    

        self.index_files = cluster_files
        print("Indexing complete.")
    

    
    def _semantic_query_ivf(self, query_vector, top_k=5, index_dir=None):
        """
        Perform a semantic query using the IVF index.

        Args:
            query_vector (np.ndarray): Query vector.
            top_k (int): Number of nearest neighbors to return.
            index_dir (str): Directory containing index files.

        Returns:
            List[Tuple[int, float]]: List of (vector_index, similarity) tuples.
        """
        query_vector = query_vector.squeeze()
        if index_dir is None or self.centroids is None:
            raise ValueError("Index must be built before querying.")

        print("Calculating nearest clusters...")
        cluster_distances = np.linalg.norm(self.centroids - query_vector, axis=1)
        nearest_clusters = np.argsort(cluster_distances)[:self.n_probe]

        candidates = []
        for cluster_id in nearest_clusters:
            cluster_path = os.path.join(index_dir, f"cluster{cluster_id}.bin")
            if not os.path.exists(cluster_path):
                continue

            
            with open(cluster_path, "rb") as f:
            # Each chunk is: 4 bytes for ID + 70 * 4 bytes for vector
                chunk_size = 4 + (70 * 4)
                
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    
                    if len(chunk) < chunk_size:
                        break  # Incomplete chunk, stop reading
                    
                    # Extract ID (first 4 bytes)
                    id = struct.unpack('I', chunk[:4])[0]
                    
                    # Extract vector (next 70 * 4 bytes)
                    vector = np.array(struct.unpack('70f', chunk[4:]))
                    
                    similarity = np.dot(query_vector, vector) / (
                        np.linalg.norm(query_vector) * np.linalg.norm(vector)
                    )
                    
                    candidates.append((id, similarity))

        print("Selecting top-k candidates...")
        # Return the IDs of the vectors with the biggest similarity
        top_candidates = nlargest(top_k, candidates, key=lambda x: x[1])
        top_candidates = [candidate[0] for candidate in top_candidates]
        return top_candidates
    
    
    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
        vectors = [(idx,vector) for idx, vector in enumerate(vectors)]
        self._IVF_index(vectors, self.index_path)
  #self._# build_index()

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
        #TODO: might change to call insert in the index, if you need
        self._# build_index()

    def get_one_row(self, row_num: int) -> np.ndarray:
        # This function is only load one row in memory
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


