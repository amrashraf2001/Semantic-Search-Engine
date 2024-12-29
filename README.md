# Semantic Search Engine with Vectorized Databases
This repository contains the code and documentation for a simple semantic search engine with vectorized databases and the evaluation of its performance. The project focuses on building an efficient indexing system to retrieve information based on vector space embeddings.

# Detailed Explanation 
## Building the Index 
• Clustering with K-Means: 
• A sample of the vectors (up to 1,000,000) is used to train a K-Means model, which 
helps partition the dataset into nlist clusters. 
• The centroids from K-Means represent cluster centers and are saved for later use 
during queries. 
• Vectors are normalized to unit length to facilitate cosine similarity calculations. 
• Assigning Vectors to Clusters: 
• Vectors are assigned to the nearest clusters based on cosine similarity with the 
centroids. 
• Each cluster's indices are saved in separate files (cluster_*.indices) within 
the index_dir directory. 
• Storing Centroids: 
• The centroids are saved in a NumPy file (centroids.npy) to be quickly loaded during 
the retrieval process. 
  
 
## Retrieval Process 
• Query Handling: 
• A query vector is normalized and compared against the centroids to determine the 
nearest clusters. 
• The system uses cosine similarity to find the closest n_probe clusters, focusing the 
search on the most relevant partitions of the dataset. 
• Candidate Selection and Similarity Computation: 
• Candidate vectors from the selected clusters are loaded in batches to manage memory 
usage effectively. 
• Cosine similarities between the query vector and candidate vectors are computed. 
• A min-heap is used to keep track of the top k candidates with the highest similarity 
scores. 
• Result Compilation: 
• The indices of the top candidates are returned as the result of the retrieval process. 
