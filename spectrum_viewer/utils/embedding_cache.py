"""Embedding computation and caching for dimensionality reduction."""

import hashlib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


class EmbeddingCache:
    """Manages embedding computation and caching for spectra data."""

    def __init__(self):
        self._cache = {}
        self.embedding_result = None
        self.pca_model = None
        self.pca_loadings = None
        self.pca_explained_var = None

    def clear(self):
        """Clear all cached embeddings."""
        self._cache.clear()
        self.embedding_result = None
        self.pca_model = None
        self.pca_loadings = None
        self.pca_explained_var = None

    def compute_embedding(self, data_objects, method="PCA", dim=2,
                          indices=None, n_pcs_for_loading=5, force=False):
        """
        Compute embedding for a subset of points (indices). If indices is None, use all points.
        Stores result as a full-size array (N_total x emb_dim) with NaNs
        for points not included in the fit, so the rest of the GUI can still index by global idx.

        Args:
            data_objects: List of Spectrum objects
            method: "PCA", "T-SNE", or "UMAP"
            dim: 2 or 3 dimensions
            indices: List of indices to include in fit, or None for all
            n_pcs_for_loading: Number of PCA components to compute for loadings plot
            force: If True, bypass cache

        Returns:
            tuple: (embedding_result, pca_model, pca_loadings, pca_explained_var)
        """
        N_total = len(data_objects)
        if N_total == 0:
            self.embedding_result = np.array([])
            return self.embedding_result, None, None, None

        if indices is None:
            fit_indices = list(range(N_total))
        else:
            fit_indices = list(indices)

        if len(fit_indices) == 0:
            # nothing to fit; keep empty embedding
            self.embedding_result = np.full((N_total, dim), np.nan, dtype=float)
            if method != "PCA":
                self.pca_model = None
                self.pca_loadings = None
                self.pca_explained_var = None
            return self.embedding_result, self.pca_model, self.pca_loadings, self.pca_explained_var

        # Build cache key that depends on the subset
        idx_bytes = np.asarray(fit_indices, dtype=np.int32).tobytes()
        idx_hash = hashlib.md5(idx_bytes).hexdigest()
        cache_key = (
            method,
            dim,
            int(n_pcs_for_loading) if method == "PCA" else None,
            idx_hash,
        )

        if (not force) and cache_key in self._cache:
            cached = self._cache[cache_key]
            if method == "PCA":
                self.embedding_result, self.pca_model, self.pca_loadings, self.pca_explained_var = cached
            else:
                self.embedding_result = cached
            return self.embedding_result, self.pca_model, self.pca_loadings, self.pca_explained_var

        # Make matrix for ONLY the visible indices
        data_matrix = np.array([data_objects[i].intensity for i in fit_indices])
        if data_matrix.size == 0:
            self.embedding_result = np.full((N_total, dim), np.nan, dtype=float)
            return self.embedding_result, None, None, None

        n_samples, n_features = data_matrix.shape
        emb_components = min(dim, n_samples, n_features)

        # PCA can fit more components than scatter dim for loadings
        pca_fit_components = min(
            max(emb_components, int(n_pcs_for_loading)),
            n_samples,
            n_features
        )

        # Clear PCA state when leaving PCA
        if method != "PCA":
            self.pca_model = None
            self.pca_loadings = None
            self.pca_explained_var = None

        try:
            if method == "T-SNE":
                # Barnes-Hut TSNE only supports n_components <= 3
                emb_components = min(emb_components, 3)

                if n_samples < emb_components + 1:
                    reducer = PCA(n_components=emb_components)
                    sub_emb = reducer.fit_transform(data_matrix)
                else:
                    perplexity = min(30, (n_samples - 1) / 3)
                    reducer = TSNE(
                        n_components=emb_components,
                        perplexity=perplexity,
                        random_state=42
                    )
                    sub_emb = reducer.fit_transform(data_matrix)

            elif method == "UMAP":
                reducer = umap.UMAP(n_components=emb_components, random_state=42)
                sub_emb = reducer.fit_transform(data_matrix)

            elif method == "PCA":
                reducer = PCA(n_components=pca_fit_components)
                full_scores = reducer.fit_transform(data_matrix)

                # Scatter uses first emb_components
                sub_emb = full_scores[:, :emb_components]

                self.pca_model = reducer
                self.pca_loadings = reducer.components_
                self.pca_explained_var = reducer.explained_variance_ratio_

            else:
                print(f"Unknown embedding method: {method}")
                self.embedding_result = np.full((N_total, dim), np.nan, dtype=float)
                return self.embedding_result, None, None, None

            # Write into full-size embedding array
            full = np.full((N_total, emb_components), np.nan, dtype=float)
            full[np.asarray(fit_indices, dtype=int), :] = sub_emb
            self.embedding_result = full

            # Cache (store PCA state too)
            if method == "PCA":
                self._cache[cache_key] = (
                    self.embedding_result, self.pca_model, self.pca_loadings, self.pca_explained_var
                )
            else:
                self._cache[cache_key] = self.embedding_result

            print(f"Computed {method} embedding on {len(fit_indices)}/{N_total} points: {sub_emb.shape}")

        except Exception as e:
            print(f"Error computing embedding: {e}")
            import traceback
            traceback.print_exc()
            self.embedding_result = np.full((N_total, dim), np.nan, dtype=float)

        return self.embedding_result, self.pca_model, self.pca_loadings, self.pca_explained_var
