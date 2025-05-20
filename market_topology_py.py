"""
Market Topology Analysis Module

This module implements computational geometry techniques for market structure analysis,
including Delaunay triangulation, Voronoi diagrams, and network analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score
from matplotlib.tri import Triangulation
import warnings

warnings.filterwarnings('ignore')


class MarketTopologyAnalyzer:
    """
    Analyze market topology using computational geometry techniques
    
    This class applies Delaunay triangulation and Voronoi diagrams to map out
    the structure of a financial market, identifying clusters, central assets,
    and relationships between assets.
    """
    
    def __init__(self, returns):
        """
        Initialize with return data
        
        Parameters:
        -----------
        returns : pandas.DataFrame
            DataFrame containing asset returns
        """
        self.returns = returns
        self.correlation_matrix = returns.corr()
        self.distance_matrix = self._calculate_distance_matrix()
        self.tickers = returns.columns.tolist()
        self.embedding = None
        self.delaunay = None
        self.voronoi = None
        self.mst = None
        self.communities = None
        
    def _calculate_distance_matrix(self):
        """
        Calculate distance matrix from correlation matrix
        
        Uses the formula d_ij = sqrt(2 * (1 - ρ_ij)) where ρ_ij is the correlation
        between assets i and j. This distance has useful mathematical properties for
        financial networks.
        """
        distance_matrix = np.sqrt(2 * (1 - self.correlation_matrix))
        return distance_matrix
    
    def compute_embedding(self, method='pca', n_components=2, perplexity=30):
        """
        Compute low-dimensional embedding from distance matrix
        
        Maps the high-dimensional distance matrix to 2D for visualization and analysis.
        
        Parameters:
        -----------
        method : str
            Embedding method: 'pca' or 'tsne'
        n_components : int
            Number of components in the embedding
        perplexity : int
            Perplexity parameter for t-SNE
            
        Returns:
        --------
        numpy.ndarray
            Embedding coordinates
        """
        # Compute embedding
        if method == 'pca':
            # PCA is faster but may not preserve local structure as well
            pca = PCA(n_components=n_components)
            self.embedding = pca.fit_transform(self.distance_matrix)
        elif method == 'tsne':
            # t-SNE preserves local structure better but is slower
            tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                        metric='precomputed', random_state=42)
            self.embedding = tsne.fit_transform(self.distance_matrix)
        else:
            raise ValueError(f"Unknown embedding method: {method}")
            
        return self.embedding
    
    def compute_delaunay(self):
        """
        Compute Delaunay triangulation
        
        Delaunay triangulation connects points such that no point is inside the
        circumcircle of any triangle. This helps identify natural neighbors and
        local structure in the market.
        
        Returns:
        --------
        scipy.spatial.Delaunay
            Delaunay triangulation object
        """
        if self.embedding is None:
            self.compute_embedding()
            
        self.delaunay = Delaunay(self.embedding)
        return self.delaunay
    
    def compute_voronoi(self):
        """
        Compute Voronoi diagram
        
        Voronoi diagrams partition the space such that each region contains all points
        closest to a particular asset. This helps identify market segmentation.
        
        Returns:
        --------
        scipy.spatial.Voronoi
            Voronoi diagram object
        """
        if self.embedding is None:
            self.compute_embedding()
            
        self.voronoi = Voronoi(self.embedding)
        return self.voronoi
    
    def compute_mst(self):
        """
        Compute Minimum Spanning Tree
        
        The MST connects all assets with minimum total distance, which captures
        the core structure of the market without redundant connections.
        
        Returns:
        --------
        networkx.Graph
            Minimum Spanning Tree as a networkx graph
        """
        # Create a complete graph with distances as weights
        G = nx.Graph()
        
        # Add nodes
        for i, ticker in enumerate(self.tickers):
            G.add_node(i, name=ticker)
        
        # Add edges with distances as weights
        for i in range(len(self.tickers)):
            for j in range(i+1, len(self.tickers)):
                G.add_edge(i, j, weight=self.distance_matrix.iloc[i, j])
        
        # Compute the minimum spanning tree
        self.mst = nx.minimum_spanning_tree(G)
        
        return self.mst
    
    def identify_communities(self, method='louvain', resolution=1.0):
        """
        Identify communities in the market graph
        
        Communities represent groups of assets that are more closely related to
        each other than to the rest of the market.
        
        Parameters:
        -----------
        method : str
            Community detection method: 'louvain', 'greedy', or 'spectral'
        resolution : float
            Resolution parameter for Louvain method
            
        Returns:
        --------
        dict
            Dictionary mapping node indices to communities
        """
        if self.mst is None:
            self.compute_mst()
            
        if method == 'louvain':
            try:
                from community import best_partition
                self.communities = best_partition(self.mst, resolution=resolution)
            except ImportError:
                print("python-louvain package not found. Using greedy modularity instead.")
                self.communities = self._greedy_modularity_communities()
        elif method == 'greedy':
            self.communities = self._greedy_modularity_communities()
        elif method == 'spectral':
            self.communities = self._spectral_communities()
        else:
            raise ValueError(f"Unknown community detection method: {method}")
            
        return self.communities
    
    def _greedy_modularity_communities(self):
        """
        Use greedy modularity maximization to find communities
        
        A fallback method if the python-louvain package is not available.
        """
        communities_generator = nx.algorithms.community.greedy_modularity_communities(self.mst)
        communities = {}
        
        for i, community in enumerate(communities_generator):
            for node in community:
                communities[node] = i
                
        return communities
    
    def _spectral_communities(self, n_clusters=None):
        """
        Use spectral clustering to find communities
        
        Spectral clustering uses the eigenvalues of the graph Laplacian to find
        natural divisions in the network.
        """
        if n_clusters is None:
            # Estimate the number of clusters using the eigenvalues of the Laplacian
            laplacian = nx.normalized_laplacian_matrix(self.mst).todense()
            eigvals = np.linalg.eigvalsh(laplacian)
            n_clusters = sum(eigvals < 1e-10) + 1
            
        adj_matrix = nx.to_numpy_array(self.mst)
        sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', 
                              random_state=42, assign_labels='kmeans')
        labels = sc.fit_predict(adj_matrix)
        
        return {i: label for i, label in enumerate(labels)}
    
    def get_asset_communities(self):
        """
        Get a mapping of assets to their communities
        
        Returns:
        --------
        dict
            Dictionary mapping asset names to community labels
        """
        if self.communities is None:
            self.identify_communities()
            
        asset_communities = {}
        for i, ticker in enumerate(self.tickers):
            asset_communities[ticker] = self.communities[i]
            
        return asset_communities
    
    def identify_central_assets(self, centrality_measure='degree', top_n=5):
        """
        Identify central assets in the market structure
        
        Central assets have a significant influence on the overall market and
        can be key drivers of market movements.
        
        Parameters:
        -----------
        centrality_measure : str
            Centrality measure to use: 'degree', 'betweenness', 'closeness', 'eigenvector'
        top_n : int
            Number of top central assets to return
            
        Returns:
        --------
        list
            List of (ticker, centrality_value) tuples for the top central assets
        """
        if self.mst is None:
            self.compute_mst()
            
        if centrality_measure == 'degree':
            # Number of connections
            centrality = nx.degree_centrality(self.mst)
        elif centrality_measure == 'betweenness':
            # Frequency of being on shortest paths
            centrality = nx.betweenness_centrality(self.mst)
        elif centrality_measure == 'closeness':
            # Inverse of average distance to all other nodes
            centrality = nx.closeness_centrality(self.mst)
        elif centrality_measure == 'eigenvector':
            # Influence based on connections to other influential nodes
            centrality = nx.eigenvector_centrality(self.mst)
        else:
            raise ValueError(f"Unknown centrality measure: {centrality_measure}")
            
        # Convert to list of (ticker, value) tuples and sort
        centrality_list = [(self.tickers[i], value) for i, value in centrality.items()]
        centrality_list.sort(key=lambda x: x[1], reverse=True)
        
        return centrality_list[:top_n]
    
    def find_market_clusters(self, n_clusters=None, method='kmeans'):
        """
        Find market clusters using the embedding
        
        Uses machine learning clustering algorithms to find natural groups in the market.
        
        Parameters:
        -----------
        n_clusters : int, optional
            Number of clusters, if None will be estimated
        method : str
            Clustering method: 'kmeans', 'dbscan', or 'spectral'
            
        Returns:
        --------
        numpy.ndarray
            Cluster labels for each asset
        """
        if self.embedding is None:
            self.compute_embedding()
            
        # Estimate number of clusters if not provided
        if n_clusters is None:
            # Try a range of clusters and pick the one with the best silhouette score
            best_score = -1
            best_n = 2
            
            for n in range(2, min(10, len(self.tickers) // 2)):
                kmeans = KMeans(n_clusters=n, random_state=42)
                labels = kmeans.fit_predict(self.embedding)
                score = silhouette_score(self.embedding, labels)
                
                if score > best_score:
                    best_score = score
                    best_n = n
                    
            n_clusters = best_n
        
        # Apply clustering method
        if method == 'kmeans':
            cluster = KMeans(n_clusters=n_clusters, random_state=42)
            labels = cluster.fit_predict(self.embedding)
        elif method == 'dbscan':
            # Estimate eps using nearest neighbor distances
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=2)
            nn.fit(self.embedding)
            distances, _ = nn.kneighbors(self.embedding)
            distances = np.sort(distances[:, 1])
            eps = np.mean(distances)
            
            cluster = DBSCAN(eps=eps, min_samples=2)
            labels = cluster.fit_predict(self.embedding)
        elif method == 'spectral':
            cluster = SpectralClustering(n_clusters=n_clusters, random_state=42)
            labels = cluster.fit_predict(self.embedding)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
            
        return labels
    
    def visualize_delaunay(self, figsize=(12, 10), save_path=None):
        """
        Visualize Delaunay triangulation
        
        Creates a plot showing the Delaunay triangulation of the market structure.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if self.delaunay is None:
            self.compute_delaunay()
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot Delaunay triangulation
        tri = Triangulation(self.embedding[:, 0], self.embedding[:, 1], 
                           self.delaunay.simplices)
        ax.triplot(tri, 'b-', alpha=0.5)
        
        # Plot points
        ax.scatter(self.embedding[:, 0], self.embedding[:, 1], c='r', s=50)
        
        # Add labels
        for i, ticker in enumerate(self.tickers):
            ax.annotate(ticker, (self.embedding[i, 0], self.embedding[i, 1]),
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_title('Market Structure: Delaunay Triangulation')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def visualize_voronoi(self, figsize=(12, 10), save_path=None):
        """
        Visualize Voronoi diagram
        
        Creates a plot showing the Voronoi diagram of the market structure.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if self.voronoi is None:
            self.compute_voronoi()
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot Voronoi diagram
        voronoi_plot_2d(self.voronoi, ax=ax, show_vertices=False,
                      line_colors='blue', line_alpha=0.5)
        
        # Plot points
        ax.scatter(self.embedding[:, 0], self.embedding[:, 1], c='r', s=50)
        
        # Add labels
        for i, ticker in enumerate(self.tickers):
            ax.annotate(ticker, (self.embedding[i, 0], self.embedding[i, 1]),
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_title('Market Structure: Voronoi Diagram')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def visualize_mst(self, figsize=(12, 10), with_communities=True, save_path=None):
        """
        Visualize Minimum Spanning Tree
        
        Creates a plot showing the minimum spanning tree of the market structure.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        with_communities : bool
            Whether to color nodes by community
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if self.mst is None:
            self.compute_mst()
            
        if with_communities and self.communities is None:
            self.identify_communities()
            
        # Prepare positions from embedding
        if self.embedding is None:
            self.compute_embedding()
            
        pos = {i: (self.embedding[i, 0], self.embedding[i, 1]) for i in range(len(self.tickers))}
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Draw nodes
        if with_communities:
            # Color nodes by community
            community_colors = {}
            for node, community in self.communities.items():
                if community not in community_colors:
                    community_colors[community] = np.random.rand(3,)
                    
            node_colors = [community_colors[self.communities[node]] for node in self.mst.nodes()]
            nx.draw_networkx_nodes(self.mst, pos, node_color=node_colors, node_size=100, ax=ax)
        else:
            nx.draw_networkx_nodes(self.mst, pos, node_color='red', node_size=100, ax=ax)
        
        # Draw edges
        nx.draw_networkx_edges(self.mst, pos, width=1.0, alpha=0.5, ax=ax)
        
        # Draw labels
        labels = {i: self.tickers[i] for i in range(len(self.tickers))}
        nx.draw_networkx_labels(self.mst, pos, labels, font_size=8, ax=ax)
        
        ax.set_title('Market Structure: Minimum Spanning Tree')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_clusters(self, labels=None, figsize=(12, 10), save_path=None):
        """
        Plot asset clusters
        
        Creates a plot showing the clusters in the market structure.
        
        Parameters:
        -----------
        labels : numpy.ndarray, optional
            Cluster labels, if None will be computed
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if self.embedding is None:
            self.compute_embedding()
            
        if labels is None:
            labels = self.find_market_clusters()
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot points colored by cluster
        scatter = ax.scatter(self.embedding[:, 0], self.embedding[:, 1], 
                           c=labels, cmap='tab10', s=50)
        
        # Add labels
        for i, ticker in enumerate(self.tickers):
            ax.annotate(ticker, (self.embedding[i, 0], self.embedding[i, 1]),
                       xytext=(5, 5), textcoords='offset points')
        
        # Add legend
        legend1 = ax.legend(*scatter.legend_elements(),
                          title="Clusters")
        ax.add_artist(legend1)
        
        ax.set_title('Market Structure: Asset Clusters')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_correlation_network(self, threshold=0.5, figsize=(12, 10), save_path=None):
        """
        Plot correlation network
        
        Creates a plot showing the correlation network of the market structure.
        
        Parameters:
        -----------
        threshold : float
            Correlation threshold for drawing edges
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        # Create correlation network
        G = nx.Graph()
        
        # Add nodes
        for i, ticker in enumerate(self.tickers):
            G.add_node(i, name=ticker)
        
        # Add edges for correlations above threshold
        for i in range(len(self.tickers)):
            for j in range(i+1, len(self.tickers)):
                corr = self.correlation_matrix.iloc[i, j]
                if abs(corr) >= threshold:
                    G.add_edge(i, j, weight=corr)
        
        # Prepare positions from embedding
        if self.embedding is None:
            self.compute_embedding()
            
        pos = {i: (self.embedding[i, 0], self.embedding[i, 1]) for i in range(len(self.tickers))}
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=100, ax=ax)
        
        # Draw edges with colors based on correlation sign
        edges_pos = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0]
        edges_neg = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 0]
        
        # Width proportional to correlation strength
        widths_pos = [2 * G[u][v]['weight'] for (u, v) in edges_pos]
        widths_neg = [2 * abs(G[u][v]['weight']) for (u, v) in edges_neg]
        
        nx.draw_networkx_edges(G, pos, edgelist=edges_pos, width=widths_pos, 
                             edge_color='green', alpha=0.5, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=edges_neg, width=widths_neg, 
                             edge_color='red', alpha=0.5, style='dashed', ax=ax)
        
        # Draw labels
        labels = {i: self.tickers[i] for i in range(len(self.tickers))}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
        
        ax.set_title(f'Market Correlation Network (threshold = {threshold})')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig