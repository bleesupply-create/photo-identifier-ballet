"""Phase 3: Face clustering using FAISS nearest-neighbor search and Chinese Whispers.

Loads face embeddings from the database, builds a similarity graph via FAISS,
then applies a two-pass Chinese Whispers algorithm (tight then loose threshold)
to group faces into identity clusters.

Usage:
    python3 cluster.py
"""

import collections
import time

import faiss
import networkx as nx
import numpy as np
from tqdm import tqdm

import config
import db


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _blob_to_embedding(blob: bytes) -> np.ndarray:
    """Deserialize a raw BLOB back into a float32 vector."""
    return np.frombuffer(blob, dtype=np.float32)


def load_embeddings():
    """Load all face embeddings and metadata from the database.

    Returns
    -------
    face_ids : list[int]
        Database primary keys for each face.
    embeddings : np.ndarray, shape (N, EMBEDDING_DIM)
        Row-aligned embedding matrix (float32, contiguous).
    detection_scores : np.ndarray, shape (N,)
        Per-face detection confidence from the detector.
    """
    conn = db.get_connection()
    rows = conn.execute(
        "SELECT id, embedding, detection_score FROM faces ORDER BY id"
    ).fetchall()
    conn.close()

    if not rows:
        return [], np.empty((0, config.EMBEDDING_DIM), dtype=np.float32), np.empty(0)

    face_ids = []
    embeddings = []
    detection_scores = []

    for row in rows:
        face_ids.append(row["id"])
        embeddings.append(_blob_to_embedding(row["embedding"]))
        detection_scores.append(row["detection_score"] or 0.0)

    embeddings = np.array(embeddings, dtype=np.float32)
    detection_scores = np.array(detection_scores, dtype=np.float32)

    # Sanity check
    assert embeddings.shape == (len(face_ids), config.EMBEDDING_DIM), (
        f"Unexpected embedding shape {embeddings.shape}; "
        f"expected ({len(face_ids)}, {config.EMBEDDING_DIM})"
    )

    return face_ids, embeddings, detection_scores


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build a FAISS L2 index over *embeddings*.

    Uses IndexIVFFlat for datasets larger than 10 000 faces (faster approximate
    search) and a brute-force IndexFlatL2 for smaller sets.
    """
    n, dim = embeddings.shape

    if n > 10_000:
        # IVF with sqrt(n) centroids is a reasonable default.
        n_cells = int(np.sqrt(n))
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, n_cells)
        index.nprobe = config.FAISS_NPROBE
        print(f"Training IVF index with {n_cells} cells ...")
        index.train(embeddings)
        index.add(embeddings)
    else:
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

    return index


def knn_search(index: faiss.Index, embeddings: np.ndarray, k: int):
    """Return (distances, indices) for a k-NN search of every embedding.

    *k* is clamped to the number of available vectors so we never ask FAISS
    for more neighbors than exist.
    """
    k = min(k, embeddings.shape[0])
    distances, indices = index.search(embeddings, k)
    return distances, indices


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(
    face_ids: list,
    distances: np.ndarray,
    indices: np.ndarray,
    threshold: float,
) -> nx.Graph:
    """Build a sparse undirected graph from k-NN results.

    An edge (i, j) is added when the L2 distance between face *i* and its
    neighbor *j* is strictly below *threshold*.  Self-loops are skipped.
    """
    G = nx.Graph()
    G.add_nodes_from(range(len(face_ids)))

    n = len(face_ids)
    for i in tqdm(range(n), desc="Building graph", unit="face"):
        for rank in range(distances.shape[1]):
            j = int(indices[i, rank])
            if j == i or j < 0:
                continue
            dist = float(distances[i, rank])
            if dist < threshold:
                # Store the tighter (smaller) distance if edge already exists.
                if G.has_edge(i, j):
                    existing = G[i][j]["weight"]
                    if dist < existing:
                        G[i][j]["weight"] = dist
                else:
                    G.add_edge(i, j, weight=dist)

    return G


# ---------------------------------------------------------------------------
# Chinese Whispers clustering
# ---------------------------------------------------------------------------

def chinese_whispers(G: nx.Graph, max_iterations: int = 100) -> dict:
    """Run the Chinese Whispers algorithm on graph *G*.

    Each node starts with a unique label.  On every iteration the nodes are
    visited in random order; each node adopts the label that appears most
    frequently among its neighbors (weighted by inverse distance so that
    closer faces have stronger influence).  The process repeats until
    convergence or *max_iterations* is reached.

    Returns
    -------
    labels : dict[int, int]
        Mapping from node index to cluster label.
    """
    nodes = list(G.nodes())
    if not nodes:
        return {}

    # Initialise: every node gets its own label.
    labels = {n: n for n in nodes}

    rng = np.random.default_rng(seed=42)

    for iteration in range(1, max_iterations + 1):
        order = list(nodes)
        rng.shuffle(order)

        changed = 0
        for node in order:
            neighbors = list(G.neighbors(node))
            if not neighbors:
                continue

            # Accumulate weighted votes for each label.
            # Weight = 1 / (distance + epsilon) so closer neighbors matter more.
            vote = collections.defaultdict(float)
            for nbr in neighbors:
                w = G[node][nbr].get("weight", 1.0)
                vote[labels[nbr]] += 1.0 / (w + 1e-6)

            best_label = max(vote, key=vote.get)
            if labels[node] != best_label:
                labels[node] = best_label
                changed += 1

        if changed == 0:
            print(f"  Chinese Whispers converged at iteration {iteration}.")
            break
    else:
        print(f"  Chinese Whispers reached max iterations ({max_iterations}).")

    return labels


# ---------------------------------------------------------------------------
# Two-pass clustering
# ---------------------------------------------------------------------------

def cluster_faces(
    face_ids: list,
    embeddings: np.ndarray,
    detection_scores: np.ndarray,
) -> dict:
    """Two-pass clustering following Apple's tight-then-loose strategy.

    Pass 1 uses a tight distance threshold to form high-precision micro-
    clusters.  Pass 2 relaxes the threshold to merge similar micro-clusters
    into final identity groups.

    Returns
    -------
    face_to_cluster : dict[int, int]
        Mapping from face database ID to final cluster label (an arbitrary
        but consistent integer).
    """
    n = len(face_ids)
    print(f"\nClustering {n} faces ...")

    # -- Build FAISS index --------------------------------------------------
    print("Building FAISS index ...")
    index = build_faiss_index(embeddings)
    k = min(config.KNN_K, n)
    print(f"Running {k}-NN search ...")
    distances, indices = knn_search(index, embeddings, k)

    # -- Pass 1: tight threshold -------------------------------------------
    print(f"\n--- Pass 1: tight threshold = {config.CLUSTER_THRESHOLD_TIGHT} ---")
    G_tight = build_graph(face_ids, distances, indices, config.CLUSTER_THRESHOLD_TIGHT)
    print(f"  Graph has {G_tight.number_of_nodes()} nodes, {G_tight.number_of_edges()} edges")
    labels_tight = chinese_whispers(G_tight)

    # Map node indices to preliminary cluster IDs.
    micro_clusters = collections.defaultdict(list)
    for node_idx, label in labels_tight.items():
        micro_clusters[label].append(node_idx)

    print(f"  Pass 1 produced {len(micro_clusters)} micro-clusters.")

    # -- Pass 2: loose threshold to merge micro-clusters --------------------
    print(f"\n--- Pass 2: loose threshold = {config.CLUSTER_THRESHOLD_LOOSE} ---")

    # Compute centroid embedding for each micro-cluster.
    mc_labels = sorted(micro_clusters.keys())
    mc_centroids = np.zeros((len(mc_labels), config.EMBEDDING_DIM), dtype=np.float32)
    mc_label_to_idx = {}
    for idx, mc_label in enumerate(mc_labels):
        member_indices = micro_clusters[mc_label]
        mc_centroids[idx] = embeddings[member_indices].mean(axis=0)
        mc_label_to_idx[mc_label] = idx

    # Build a second FAISS index over centroids.
    mc_index = build_faiss_index(mc_centroids)
    mc_k = min(config.KNN_K, len(mc_labels))
    mc_distances, mc_indices = knn_search(mc_index, mc_centroids, mc_k)

    G_loose = build_graph(
        mc_labels, mc_distances, mc_indices, config.CLUSTER_THRESHOLD_LOOSE,
    )
    print(f"  Graph has {G_loose.number_of_nodes()} nodes, {G_loose.number_of_edges()} edges")
    merge_labels = chinese_whispers(G_loose)

    # -- Build final mapping: face_id -> cluster_label ----------------------
    # Determine which micro-clusters merged.
    final_cluster_map = {}  # mc_label -> final_cluster
    for mc_idx, final_label in merge_labels.items():
        mc_label = mc_labels[mc_idx]
        final_cluster_map[mc_label] = final_label

    # Assign contiguous cluster IDs starting from 1.
    unique_final = sorted(set(final_cluster_map.values()))
    relabel = {old: new_id for new_id, old in enumerate(unique_final, start=1)}

    face_to_cluster = {}
    for mc_label, members in micro_clusters.items():
        final = relabel[final_cluster_map[mc_label]]
        for node_idx in members:
            face_to_cluster[face_ids[node_idx]] = final

    return face_to_cluster


# ---------------------------------------------------------------------------
# Database write-back
# ---------------------------------------------------------------------------

def write_clusters_to_db(
    face_to_cluster: dict,
    detection_scores_map: dict,
):
    """Persist cluster assignments to the database.

    - Updates ``faces.cluster_id`` for every face.
    - Rebuilds the ``clusters`` table with ``face_count`` and the
      ``representative_face_id`` (the face with the highest detection score
      in each cluster).
    """
    conn = db.get_connection()

    # -- Update faces table -------------------------------------------------
    print("\nWriting cluster IDs to faces table ...")
    for face_id, cluster_id in tqdm(face_to_cluster.items(), desc="Updating faces", unit="face"):
        conn.execute(
            "UPDATE faces SET cluster_id = ? WHERE id = ?",
            (cluster_id, face_id),
        )
    conn.commit()

    # -- Rebuild clusters table ---------------------------------------------
    print("Rebuilding clusters table ...")

    # Cache existing dancer_name mappings BEFORE deleting rows.
    existing_names = {}
    for row in conn.execute("SELECT id, dancer_name FROM clusters WHERE dancer_name IS NOT NULL"):
        existing_names[row["id"]] = row["dancer_name"]

    conn.execute("DELETE FROM clusters")

    # Gather cluster members.
    cluster_members = collections.defaultdict(list)
    for face_id, cluster_id in face_to_cluster.items():
        cluster_members[cluster_id].append(face_id)

    for cluster_id in tqdm(sorted(cluster_members), desc="Writing clusters", unit="cluster"):
        members = cluster_members[cluster_id]
        face_count = len(members)

        # Representative = highest detection_score.
        representative = max(members, key=lambda fid: detection_scores_map.get(fid, 0.0))

        # Preserve any existing dancer_name for this cluster id.
        dancer_name = existing_names.get(cluster_id)

        conn.execute(
            """INSERT OR REPLACE INTO clusters (id, dancer_name, face_count, representative_face_id)
               VALUES (?, ?, ?, ?)""",
            (cluster_id, dancer_name, face_count, representative),
        )

    conn.commit()
    conn.close()

    print(f"  Wrote {len(cluster_members)} clusters to database.")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(face_to_cluster: dict):
    """Print a human-readable summary of the clustering results."""
    total_faces = len(face_to_cluster)
    cluster_sizes = collections.Counter(face_to_cluster.values())
    num_clusters = len(cluster_sizes)
    largest = max(cluster_sizes.values()) if cluster_sizes else 0
    singletons = sum(1 for c in cluster_sizes.values() if c == 1)

    print("\n" + "=" * 50)
    print("  Clustering Summary")
    print("=" * 50)
    print(f"  Total faces:       {total_faces}")
    print(f"  Clusters found:    {num_clusters}")
    print(f"  Largest cluster:   {largest} faces")
    print(f"  Singletons:        {singletons}")
    print("=" * 50 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    db.init_db()

    # -- Load embeddings ----------------------------------------------------
    print("Loading face embeddings from database ...")
    face_ids, embeddings, detection_scores = load_embeddings()

    if len(face_ids) == 0:
        print("No faces found in the database. Run the detector first.")
        return

    print(f"  Loaded {len(face_ids)} faces with {config.EMBEDDING_DIM}-dim embeddings.")

    # -- Cluster ------------------------------------------------------------
    face_to_cluster = cluster_faces(face_ids, embeddings, detection_scores)

    # -- Write back ---------------------------------------------------------
    detection_scores_map = dict(zip(face_ids, detection_scores.tolist()))
    write_clusters_to_db(face_to_cluster, detection_scores_map)

    # -- Summary ------------------------------------------------------------
    print_summary(face_to_cluster)

    elapsed = time.time() - t0
    print(f"Clustering completed in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
