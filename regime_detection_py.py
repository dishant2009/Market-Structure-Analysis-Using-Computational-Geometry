"""
Market Regime Detection Module

This module implements graph-based clustering techniques to detect and analyze
different market regimes and track transitions between them.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class MarketRegimeDetector:
    """
    Detect market regimes using graph-based clustering techniques
    
    This class identifies distinct market regimes over time by analyzing the
    evolving structure of market graphs and tracking transitions between regimes.
    """
    
    def __init__(self, returns, window_size=60, step_size=20, min_window=30):
        """
        Initialize with return data
        
        Parameters:
        -----------
        returns : pandas.DataFrame
            DataFrame containing asset returns
        window_size : int
            Size of the rolling window (in trading days)
        step_size : int
            Step size for rolling window
        min_window : int
            Minimum window size for initial periods
        """
        self.returns = returns
        self.window_size = window_size
        self.step_size = step_size
        self.min_window = min_window
        self.regime_labels = None
        self.regime_periods = None
        self.regime_transitions = None
        
    def detect_regimes(self, n_regimes=None, method='graph', use_hmm=False):
        """
        Detect market regimes
        
        Parameters:
        -----------
        n_regimes : int, optional
            Number of regimes to detect, if None will be estimated
        method : str
            Method for regime detection: 'graph', 'correlation', or 'volatility'
        use_hmm : bool
            Whether to use HMM for smoothing regime transitions
            
        Returns:
        --------
        pandas.Series
            Series of regime labels indexed by time
        """
        # Initialize variables
        dates = []
        features = []
        
        print(f"Detecting market regimes using {method} method...")
        print(f"Window size: {self.window_size}, Step size: {self.step_size}")
        
        # Compute rolling window features
        for i in range(0, len(self.returns) - self.min_window, self.step_size):
            end_idx = i + self.window_size
            if end_idx > len(self.returns):
                end_idx = len(self.returns)
                
            window = self.returns.iloc[i:end_idx]
            
            if len(window) < self.min_window:
                continue
                
            # Get the end date of the window
            date = window.index[-1]
            dates.append(date)
            
            # Extract features based on the method
            if method == 'graph':
                feature = self._extract_graph_features(window)
            elif method == 'correlation':
                feature = self._extract_correlation_features(window)
            elif method == 'volatility':
                feature = self._extract_volatility_features(window)
            else:
                raise ValueError(f"Unknown method: {method}")
                
            features.append(feature)
            
        print(f"Extracted {len(features)} feature sets from windows")
        
        # Convert features to numpy array
        features = np.array(features)
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Estimate number of regimes if not provided
        if n_regimes is None:
            # Try a range of clusters and pick the one with the best silhouette score
            best_score = -1
            best_n = 2
            
            print("Estimating optimal number of regimes...")
            for n in range(2, min(10, len(features) // 5)):
                kmeans = KMeans(n_clusters=n, random_state=42)
                labels = kmeans.fit_predict(features_scaled)
                
                # Skip if only one cluster has points
                if len(np.unique(labels)) < 2:
                    continue
                    
                score = silhouette_score(features_scaled, labels)
                print(f"  n_regimes={n}, silhouette score={score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_n = n
                    
            n_regimes = best_n
            print(f"Optimal number of regimes: {n_regimes}")
        
        # Apply K-means clustering
        print(f"Clustering into {n_regimes} regimes...")
        kmeans = KMeans(n_clusters=n_regimes, random_state=42)
        labels = kmeans.fit_predict(features_scaled)
        
        # Apply HMM for smoothing if requested
        if use_hmm:
            try:
                from hmmlearn import hmm
                
                print("Applying HMM for smoothing regime transitions...")
                # Train HMM
                model = hmm.GaussianHMM(n_components=n_regimes, covariance_type="full", 
                                     n_iter=100, random_state=42)
                model.fit(features_scaled)
                
                # Decode states
                labels = model.predict(features_scaled)
            except ImportError:
                print("hmmlearn package not found. Skipping HMM smoothing.")
        
        # Create Series of regime labels
        self.regime_labels = pd.Series(labels, index=dates, name='regime')
        
        # Identify regime periods
        self.regime_periods = self._identify_regime_periods()
        
        # Identify regime transitions
        self.regime_transitions = self._identify_regime_transitions()
        
        print(f"Detected {len(self.regime_periods)} regime periods")
        
        return self.regime_labels
    
    def _extract_graph_features(self, window):
        """
        Extract graph-based features from return window
        
        Creates a correlation network and extracts topological features that
        characterize the market structure.
        """
        # Compute correlation matrix
        corr_matrix = window.corr()
        
        # Create weighted graph from correlation matrix
        G = nx.Graph()
        
        # Add nodes
        for i in range(len(corr_matrix)):
            G.add_node(i)
        
        # Add edges with correlation as weight
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                # Use absolute correlation as weight
                weight = abs(corr_matrix.iloc[i, j])
                G.add_edge(i, j, weight=weight)
        
        # Extract graph features
        features = []
        
        # Average degree
        avg_degree = np.mean([d for _, d in G.degree()])
        features.append(avg_degree)
        
        # Clustering coefficient
        clustering = nx.average_clustering(G, weight='weight')
        features.append(clustering)
        
        # Assortativity
        try:
            assortativity = nx.degree_assortativity_coefficient(G, weight='weight')
            if np.isnan(assortativity):
                assortativity = 0
        except:
            assortativity = 0
        features.append(assortativity)
        
        # Edge density
        density = nx.density(G)
        features.append(density)
        
        # Weighted average path length (for connected components)
        components = list(nx.connected_components(G))
        avg_path_lengths = []
        for comp in components:
            subgraph = G.subgraph(comp)
            if len(subgraph) > 1:
                try:
                    # Use inverse of weight as distance
                    length = nx.average_shortest_path_length(
                        subgraph, weight=lambda u, v, d: 1.0 / (d['weight'] + 1e-6))
                    avg_path_lengths.append(length)
                except:
                    pass
        
        if avg_path_lengths:
            avg_path_length = np.mean(avg_path_lengths)
        else:
            avg_path_length = 0
        features.append(avg_path_length)
        
        # Number of communities (using Louvain method)
        try:
            from community import best_partition
            partition = best_partition(G)
            n_communities = len(set(partition.values()))
        except ImportError:
            # Fall back to connected components
            n_communities = len(components)
        features.append(n_communities)
        
        # Average node centrality (various measures)
        try:
            centrality = nx.eigenvector_centrality(G, weight='weight')
            avg_centrality = np.mean(list(centrality.values()))
        except:
            avg_centrality = 0
        features.append(avg_centrality)
        
        try:
            centrality = nx.betweenness_centrality(G, weight='weight')
            avg_centrality = np.mean(list(centrality.values()))
        except:
            avg_centrality = 0
        features.append(avg_centrality)
        
        # Spectral properties
        try:
            eigenvalues = np.sort(nx.normalized_laplacian_spectrum(G))
            # Spectral gap
            spectral_gap = eigenvalues[1] - eigenvalues[0]
            features.append(spectral_gap)
            
            # Energy (sum of absolute eigenvalues)
            energy = np.sum(np.abs(eigenvalues))
            features.append(energy)
        except:
            features.extend([0, 0])  # Placeholder for failed spectral calculations
        
        return features
    
    def _extract_correlation_features(self, window):
        """
        Extract correlation-based features from return window
        
        Focuses on patterns in the correlation structure to identify regime characteristics.
        """
        # Compute correlation matrix
        corr_matrix = window.corr()
        
        # Extract features
        features = []
        
        # Average correlation
        avg_corr = np.mean(np.triu(corr_matrix.values, k=1))
        features.append(avg_corr)
        
        # Correlation dispersion (std of correlations)
        corr_std = np.std(np.triu(corr_matrix.values, k=1))
        features.append(corr_std)
        
        # Percentage of positive correlations
        triu_indices = np.triu_indices(len(corr_matrix), k=1)
        triu_values = corr_matrix.values[triu_indices]
        pct_positive = np.mean(triu_values > 0)
        features.append(pct_positive)
        
        # Principal components analysis
        pca = PCA(n_components=min(5, len(corr_matrix)-1))
        pca.fit(window)
        
        # Variance explained by first component (market factor)
        var_explained = pca.explained_variance_ratio_[0]
        features.append(var_explained)
        
        # Ratio of first to second component (market dominance)
        if len(pca.explained_variance_ratio_) > 1:
            ratio = pca.explained_variance_ratio_[0] / pca.explained_variance_ratio_[1]
        else:
            ratio = 1.0
        features.append(ratio)
        
        # Sum of first 3 components explained variance
        if len(pca.explained_variance_ratio_) >= 3:
            top3_var = np.sum(pca.explained_variance_ratio_[:3])
        else:
            top3_var = np.sum(pca.explained_variance_ratio_)
        features.append(top3_var)
        
        return features
    
    def _extract_volatility_features(self, window):
        """
        Extract volatility-based features from return window
        
        Focuses on risk and return patterns to identify regime characteristics.
        """
        # Extract features
        features = []
        
        # Average volatility
        vols = window.std()
        avg_vol = np.mean(vols)
        features.append(avg_vol)
        
        # Volatility dispersion
        vol_std = np.std(vols)
        features.append(vol_std)
        
        # Maximum drawdown
        cumulative_returns = (1 + window).cumprod()
        max_drawdowns = []
        for col in cumulative_returns:
            series = cumulative_returns[col]
            rolling_max = series.cummax()
            drawdown = (series / rolling_max) - 1
            max_drawdown = drawdown.min()
            max_drawdowns.append(max_drawdown)
        avg_max_drawdown = np.mean(max_drawdowns)
        features.append(avg_max_drawdown)
        
        # Average cross-asset correlation
        corr_matrix = window.corr()
        avg_corr = np.mean(np.triu(corr_matrix.values, k=1))
        features.append(avg_corr)
        
        # Average skewness
        skewness = window.skew()
        avg_skew = np.mean(skewness)
        features.append(avg_skew)
        
        # Average kurtosis
        kurtosis = window.kurtosis()
        avg_kurt = np.mean(kurtosis)
        features.append(avg_kurt)
        
        # VIX-like measure (for multi-asset)
        # Compute a covariance-weighted volatility measure
        cov_matrix = window.cov()
        weights = np.ones(len(cov_matrix)) / len(cov_matrix)  # Equal weights
        portfolio_var = weights.dot(cov_matrix).dot(weights)
        portfolio_vol = np.sqrt(portfolio_var)
        features.append(portfolio_vol)
        
        # Correlation-weighted volatility
        corr_matrix = window.corr()
        avg_corrs = np.mean(corr_matrix.values, axis=1)
        corr_weighted_vol = np.sum(vols * avg_corrs) / np.sum(avg_corrs)
        features.append(corr_weighted_vol)
        
        return features
    
    def _identify_regime_periods(self):
        """
        Identify contiguous time periods for each regime
        
        Groups consecutive dates with the same regime into periods.
        """
        if self.regime_labels is None:
            raise ValueError("Regimes have not been detected yet")
            
        periods = []
        
        # Sort by time
        sorted_labels = self.regime_labels.sort_index()
        
        # Get unique regimes
        regimes = sorted_labels.unique()
        
        for regime in regimes:
            # Get dates for this regime
            regime_dates = sorted_labels[sorted_labels == regime].index
            
            # Find contiguous periods
            if len(regime_dates) > 0:
                # Initialize
                start_date = regime_dates[0]
                prev_date = regime_dates[0]
                
                for i in range(1, len(regime_dates)):
                    date = regime_dates[i]
                    
                    # If there's a gap larger than the step size, this is a new period
                    if (date - prev_date).days > self.step_size * 2:
                        periods.append({
                            'regime': int(regime),
                            'start_date': start_date,
                            'end_date': prev_date,
                            'duration': (prev_date - start_date).days
                        })
                        start_date = date
                        
                    prev_date = date
                
                # Add the last period
                periods.append({
                    'regime': int(regime),
                    'start_date': start_date,
                    'end_date': prev_date,
                    'duration': (prev_date - start_date).days
                })
        
        # Convert to DataFrame and sort by start date
        periods_df = pd.DataFrame(periods)
        periods_df = periods_df.sort_values('start_date')
        
        return periods_df
    
    def _identify_regime_transitions(self):
        """
        Identify regime transitions
        
        Identifies points where the market shifts from one regime to another.
        """
        if self.regime_labels is None:
            raise ValueError("Regimes have not been detected yet")
            
        transitions = []
        
        # Sort by time
        sorted_labels = self.regime_labels.sort_index()
        
        # Find transitions
        prev_regime = sorted_labels.iloc[0]
        prev_date = sorted_labels.index[0]
        
        for i in range(1, len(sorted_labels)):
            regime = sorted_labels.iloc[i]
            date = sorted_labels.index[i]
            
            if regime != prev_regime:
                transitions.append({
                    'date': date,
                    'from_regime': int(prev_regime),
                    'to_regime': int(regime)
                })
                prev_regime = regime
                
            prev_date = date
        
        # Convert to DataFrame and sort by date
        transitions_df = pd.DataFrame(transitions)
        if not transitions_df.empty:
            transitions_df = transitions_df.sort_values('date')
        
        return transitions_df
    
    def plot_regime_labels(self, figsize=(12, 6), save_path=None):
        """
        Plot regime labels over time
        
        Creates a time series plot of regime labels.
        
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
        if self.regime_labels is None:
            raise ValueError("Regimes have not been detected yet")
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot regime labels
        self.regime_labels.plot(ax=ax, marker='o', linestyle='-')
        
        # Format y-axis to show integer labels
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        ax.set_title('Market Regimes Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Regime')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_regime_transitions(self, returns_index=None, figsize=(12, 6), save_path=None):
        """
        Plot regime transitions with market returns
        
        Creates a plot showing market returns with vertical lines at regime transitions.
        
        Parameters:
        -----------
        returns_index : pandas.Series, optional
            Series of market returns or cumulative returns for the background
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if self.regime_labels is None:
            raise ValueError("Regimes have not been detected yet")
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create cumulative returns if not provided
        if returns_index is None:
            # Use equal-weighted portfolio of returns
            portfolio_returns = self.returns.mean(axis=1)
            returns_index = (1 + portfolio_returns).cumprod()
        
        # Plot returns
        returns_index.plot(ax=ax)
        
        # Plot regime transitions as vertical lines
        if not self.regime_transitions.empty:
            for _, row in self.regime_transitions.iterrows():
                ax.axvline(x=row['date'], color='r', linestyle='--', alpha=0.5)
                ax.annotate(f"{row['from_regime']}->{row['to_regime']}", 
                           xy=(row['date'], ax.get_ylim()[1]),
                           xytext=(0, 10), textcoords='offset points',
                           ha='center', va='bottom', rotation=90)
        
        ax.set_title('Market Regimes and Transitions')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Returns')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_regime_characteristics(self, figsize=(12, 10), save_path=None):
        """
        Plot regime characteristics
        
        Creates a set of plots showing key statistical properties of each regime.
        
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
        if self.regime_labels is None:
            raise ValueError("Regimes have not been detected yet")
            
        # Get regime dates
        regime_dates = {}
        for regime in self.regime_labels.unique():
            regime_dates[regime] = self.regime_labels[self.regime_labels == regime].index
        
        # Calculate statistics for each regime
        regime_stats = {}
        
        for regime, dates in regime_dates.items():
            # Filter returns for this regime
            regime_returns = []
            
            for date in dates:
                # Find returns on this date
                if date in self.returns.index:
                    regime_returns.append(self.returns.loc[date])
                # Try to find closest date within a week
                else:
                    closest_date = self.returns.index[self.returns.index.searchsorted(date)]
                    if abs((closest_date - date).days) <= 7:
                        regime_returns.append(self.returns.loc[closest_date])
            
            if regime_returns:
                regime_returns_df = pd.concat(regime_returns, axis=1).T
                
                # Calculate statistics
                stats = {
                    'mean_return': regime_returns_df.mean().mean() * 252,  # Annualized
                    'volatility': regime_returns_df.std().mean() * np.sqrt(252),  # Annualized
                    'sharpe': (regime_returns_df.mean().mean() / regime_returns_df.std().mean()) * np.sqrt(252),
                    'skewness': regime_returns_df.skew().mean(),
                    'kurtosis': regime_returns_df.kurtosis().mean(),
                    'correlation': regime_returns_df.corr().values[np.triu_indices(len(regime_returns_df.columns), k=1)].mean(),
                    'max_drawdown': (regime_returns_df.cumsum().cummax() - regime_returns_df.cumsum()).max().mean()
                }
                
                regime_stats[regime] = stats
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        axes = axes.flatten()
        
        metrics = ['mean_return', 'volatility', 'sharpe', 'correlation', 'skewness', 'max_drawdown']
        titles = ['Annualized Return', 'Annualized Volatility', 'Sharpe Ratio', 
                 'Average Correlation', 'Skewness', 'Max Drawdown']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            values = [stats[metric] for regime, stats in regime_stats.items()]
            regimes = list(regime_stats.keys())
            
            # Sort by regime number
            sorted_idx = np.argsort(regimes)
            values = [values[i] for i in sorted_idx]
            regimes = [regimes[i] for i in sorted_idx]
            
            # Plot
            bars = axes[i].bar(regimes, values)
            
            # Change color based on value (green for good, red for bad)
            if metric in ['mean_return', 'sharpe']:
                # Higher is better
                colors = ['g' if v > 0 else 'r' for v in values]
            elif metric in ['volatility', 'max_drawdown']:
                # Lower is better
                colors = ['r' if v > 0 else 'g' for v in values]
            else:
                colors = ['b'] * len(values)
                
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            axes[i].set_title(title)
            axes[i].set_xlabel('Regime')
            
            # Format y-axis for percentages
            if metric in ['mean_return', 'volatility', 'max_drawdown']:
                axes[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig