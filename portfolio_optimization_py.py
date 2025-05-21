"""
Portfolio Optimization Module

This module implements portfolio optimization techniques that leverage computational
geometry insights to reduce risk and improve performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize
import warnings

warnings.filterwarnings('ignore')


class PortfolioOptimizer:
    """
    Portfolio optimization using computational geometry insights
    
    This class implements various portfolio optimization strategies and leverages
    the structural insights from market topology to reduce risk.
    """
    
    def __init__(self, returns, prices=None, risk_free_rate=0.0):
        """
        Initialize portfolio optimizer
        
        Parameters:
        -----------
        returns : pandas.DataFrame
            DataFrame containing asset returns
        prices : pandas.DataFrame, optional
            DataFrame containing asset prices
        risk_free_rate : float
            Risk-free rate (annualized)
        """
        self.returns = returns
        self.prices = prices
        self.risk_free_rate = risk_free_rate
        self.assets = returns.columns.tolist()
        self.n_assets = len(self.assets)
        
        # Calculate mean returns and covariance
        self.mean_returns = returns.mean() * 252  # Annualized
        self.cov_matrix = returns.cov() * 252     # Annualized
        
        print(f"Portfolio Optimizer initialized with {self.n_assets} assets")
        print(f"Risk-free rate: {self.risk_free_rate:.2%}")
        
    def optimize_portfolio(self, method='max_sharpe', target_return=None, 
                         min_weight=0.0, max_weight=1.0, use_topology=True,
                         topology=None):
        """
        Optimize portfolio weights
        
        Parameters:
        -----------
        method : str
            Optimization method: 'min_risk', 'max_sharpe', 'max_return', 'risk_parity',
            'equal_weight', or 'target_return'
        target_return : float, optional
            Target return for 'target_return' method
        min_weight : float
            Minimum weight constraint
        max_weight : float
            Maximum weight constraint
        use_topology : bool
            Whether to use topology analysis for optimization
        topology : MarketTopologyAnalyzer, optional
            Pre-computed topology analyzer
            
        Returns:
        --------
        pandas.Series
            Optimized portfolio weights
        """
        print(f"Optimizing portfolio using {method} method {'with' if use_topology else 'without'} topology insights")
        
        if use_topology:
            if topology is None:
                # If topology not provided, import and compute it
                from market_topology import MarketTopologyAnalyzer
                
                print("Computing market topology...")
                topology = MarketTopologyAnalyzer(self.returns)
                
            # Compute topology
            topology.compute_embedding()
            topology.compute_delaunay()
            topology.compute_mst()
            communities = topology.identify_communities()
            
            # Get asset communities
            asset_communities = topology.get_asset_communities()
            
            # Modify constraints based on topology
            return self._optimize_with_topology(method, target_return, min_weight, max_weight, asset_communities)
        else:
            # Standard optimization
            return self._optimize_standard(method, target_return, min_weight, max_weight)
    
    def _optimize_standard(self, method, target_return, min_weight, max_weight):
        """
        Standard portfolio optimization
        
        Implements various optimization methods without using topology information.
        """
        if method == 'min_risk':
            return self._min_risk_portfolio(min_weight, max_weight)
        elif method == 'max_sharpe':
            return self._max_sharpe_portfolio(min_weight, max_weight)
        elif method == 'max_return':
            return self._max_return_portfolio(min_weight, max_weight)
        elif method == 'risk_parity':
            return self._risk_parity_portfolio(min_weight, max_weight)
        elif method == 'equal_weight':
            return self._equal_weight_portfolio()
        elif method == 'target_return':
            if target_return is None:
                raise ValueError("target_return must be specified for 'target_return' method")
            return self._target_return_portfolio(target_return, min_weight, max_weight)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _optimize_with_topology(self, method, target_return, min_weight, max_weight, asset_communities):
        """
        Portfolio optimization using topology information
        
        Uses community structure to improve optimization by first optimizing within
        communities, then allocating across communities.
        """
        # Get community information
        community_assets = {}
        for asset, community in asset_communities.items():
            if community not in community_assets:
                community_assets[community] = []
            community_assets[community].append(asset)
        
        print(f"Optimizing within {len(community_assets)} communities...")
        
        # Optimize within each community
        community_weights = {}
        for community, assets in community_assets.items():
            # Get returns for this community
            community_returns = self.returns[assets]
            
            # Create community optimizer
            community_optimizer = PortfolioOptimizer(community_returns, risk_free_rate=self.risk_free_rate)
            
            # Optimize within community
            community_weights[community] = community_optimizer.optimize_portfolio(
                method=method, target_return=target_return, 
                min_weight=min_weight, max_weight=max_weight,
                use_topology=False
            )
        
        # Determine community allocations
        if method == 'equal_weight':
            # Equal weight across communities
            community_allocations = pd.Series({
                community: 1.0 / len(community_assets) 
                for community in community_assets
            })
        elif method == 'min_risk':
            # Risk-based community allocation
            community_vars = {}
            for community, assets in community_assets.items():
                # Get community weights
                weights = community_weights[community]
                
                # Get community covariance
                community_cov = self.cov_matrix.loc[assets, assets]
                
                # Calculate variance
                community_vars[community] = weights.dot(community_cov).dot(weights)
            
            # Inverse variance weighting
            total_inv_var = sum(1.0 / var for var in community_vars.values())
            community_allocations = pd.Series({
                community: (1.0 / var) / total_inv_var
                for community, var in community_vars.items()
            })
        else:
            # Use risk parity across communities
            community_allocations = self._risk_parity_communities(community_assets, community_weights)
        
        print("Combining community allocations...")
        
        # Combine community weights into overall portfolio
        portfolio_weights = pd.Series(0.0, index=self.assets)
        
        for community, allocation in community_allocations.items():
            assets = community_assets[community]
            weights = community_weights[community]
            
            # Apply allocation to weights
            for asset in assets:
                portfolio_weights[asset] = weights[asset] * allocation
        
        return portfolio_weights
    
    def _risk_parity_communities(self, community_assets, community_weights):
        """
        Risk parity allocation across communities
        
        Allocates to communities such that they contribute equally to portfolio risk.
        """
        # Calculate risk contribution for each community
        community_risks = {}
        for community, assets in community_assets.items():
            # Get community weights
            weights = community_weights[community]
            
            # Get community covariance
            community_cov = self.cov_matrix.loc[assets, assets]
            
            # Calculate variance
            community_var = weights.dot(community_cov).dot(weights)
            community_risks[community] = np.sqrt(community_var)
        
        # Inverse risk weighting
        total_inv_risk = sum(1.0 / risk for risk in community_risks.values())
        community_allocations = pd.Series({
            community: (1.0 / risk) / total_inv_risk
            for community, risk in community_risks.items()
        })
        
        return community_allocations
    
    def _min_risk_portfolio(self, min_weight, max_weight):
        """
        Minimum risk portfolio
        
        Finds the portfolio with the lowest volatility.
        """
        # Define objective function (portfolio variance)
        def objective(weights):
            return weights.dot(self.cov_matrix).dot(weights)
        
        # Define constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Sum of weights = 1
        ]
        
        # Define bounds
        bounds = tuple((min_weight, max_weight) for _ in range(self.n_assets))
        
        # Initial guess (equal weights)
        x0 = np.ones(self.n_assets) / self.n_assets
        
        # Optimize
        result = scipy.optimize.minimize(
            objective, x0, method='SLSQP', bounds=bounds, constraints=constraints
        )
        
        # Get optimal weights
        weights = pd.Series(result['x'], index=self.assets)
        
        return weights
    
    def _max_sharpe_portfolio(self, min_weight, max_weight):
        """
        Maximum Sharpe ratio portfolio
        
        Finds the portfolio with the highest Sharpe ratio (return per unit of risk).
        """
        # Define objective function (negative Sharpe ratio)
        def objective(weights):
            portfolio_return = np.sum(weights * self.mean_returns)
            portfolio_stddev = np.sqrt(weights.dot(self.cov_matrix).dot(weights))
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_stddev
            return -sharpe  # Negative because we're minimizing
        
        # Define constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Sum of weights = 1
        ]
        
        # Define bounds
        bounds = tuple((min_weight, max_weight) for _ in range(self.n_assets))
        
        # Initial guess (equal weights)
        x0 = np.ones(self.n_assets) / self.n_assets
        
        # Optimize
        result = scipy.optimize.minimize(
            objective, x0, method='SLSQP', bounds=bounds, constraints=constraints
        )
        
        # Get optimal weights
        weights = pd.Series(result['x'], index=self.assets)
        
        return weights
    
    def _max_return_portfolio(self, min_weight, max_weight):
        """
        Maximum return portfolio
        
        Finds the portfolio with the highest expected return.
        """
        # For maximum return, we simply allocate all to the asset with highest return
        # But we need to respect bounds
        
        # Find asset with highest return
        best_asset = self.mean_returns.idxmax()
        
        # Initialize weights
        weights = pd.Series(min_weight, index=self.assets)
        
        # Allocate maximum to best asset
        weights[best_asset] = max_weight
        
        # Adjust to ensure sum = 1
        weight_sum = weights.sum()
        if weight_sum < 1.0:
            # Sort assets by return
            sorted_assets = self.mean_returns.sort_values(ascending=False).index
            
            # Allocate remaining weight
            remaining = 1.0 - weight_sum
            for asset in sorted_assets:
                if asset == best_asset:
                    continue
                    
                # Calculate how much we can add
                add_amount = min(max_weight - weights[asset], remaining)
                weights[asset] += add_amount
                remaining -= add_amount
                
                if remaining <= 1e-10:
                    break
        
        return weights
    
    def _risk_parity_portfolio(self, min_weight, max_weight):
        """
        Risk parity portfolio
        
        Allocates such that each asset contributes equally to portfolio risk.
        """
        # Define objective function (sum of squared risk contribution differences)
        def objective(weights):
            weights = np.array(weights)
            portfolio_risk = np.sqrt(weights.dot(self.cov_matrix).dot(weights))
            risk_contribution = weights * (self.cov_matrix.dot(weights)) / portfolio_risk
            target_risk_contribution = portfolio_risk / self.n_assets
            return np.sum((risk_contribution - target_risk_contribution) ** 2)
        
        # Define constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Sum of weights = 1
        ]
        
        # Define bounds
        bounds = tuple((min_weight, max_weight) for _ in range(self.n_assets))
        
        # Initial guess (equal weights)
        x0 = np.ones(self.n_assets) / self.n_assets
        
        # Optimize
        result = scipy.optimize.minimize(
            objective, x0, method='SLSQP', bounds=bounds, constraints=constraints
        )
        
        # Get optimal weights
        weights = pd.Series(result['x'], index=self.assets)
        
        return weights
    
    def _equal_weight_portfolio(self):
        """
        Equal weight portfolio
        
        Allocates the same weight to each asset.
        """
        weights = pd.Series(1.0 / self.n_assets, index=self.assets)
        return weights
    
    def _target_return_portfolio(self, target_return, min_weight, max_weight):
        """
        Minimum risk portfolio with target return
        
        Finds the portfolio with the lowest risk that achieves a target return.
        """
        # Define objective function (portfolio variance)
        def objective(weights):
            return weights.dot(self.cov_matrix).dot(weights)
        
        # Define constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},  # Sum of weights = 1
            {'type': 'eq', 'fun': lambda x: np.sum(x * self.mean_returns) - target_return}  # Target return
        ]
        
        # Define bounds
        bounds = tuple((min_weight, max_weight) for _ in range(self.n_assets))
        
        # Initial guess (equal weights)
        x0 = np.ones(self.n_assets) / self.n_assets
        
        try:
            # Optimize
            result = scipy.optimize.minimize(
                objective, x0, method='SLSQP', bounds=bounds, constraints=constraints
            )
            
            if not result['success']:
                # Fall back to max return portfolio
                print(f"Target return {target_return:.2%} not feasible. Using max return portfolio.")
                return self._max_return_portfolio(min_weight, max_weight)
            
            # Get optimal weights
            weights = pd.Series(result['x'], index=self.assets)
        except:
            # Fall back to max return portfolio
            print(f"Target return {target_return:.2%} optimization failed. Using max return portfolio.")
            weights = self._max_return_portfolio(min_weight, max_weight)
        
        return weights
    
    def portfolio_performance(self, weights):
        """
        Calculate portfolio performance metrics
        
        Parameters:
        -----------
        weights : pandas.Series
            Portfolio weights
            
        Returns:
        --------
        dict
            Performance metrics
        """
        # Calculate expected return and risk
        expected_return = np.sum(weights * self.mean_returns)
        risk = np.sqrt(weights.dot(self.cov_matrix).dot(weights))
        
        # Calculate Sharpe ratio
        sharpe = (expected_return - self.risk_free_rate) / risk
        
        # Calculate diversification ratio
        weighted_vols = weights * np.sqrt(np.diag(self.cov_matrix))
        sum_weighted_vols = np.sum(weighted_vols)
        diversification_ratio = sum_weighted_vols / risk
        
        # Calculate risk contribution
        marginal_contrib = self.cov_matrix.dot(weights)
        risk_contrib = weights * marginal_contrib / risk
        
        # Performance dictionary
        performance = {
            'expected_return': expected_return,
            'risk': risk,
            'sharpe_ratio': sharpe,
            'diversification_ratio': diversification_ratio,
            'risk_contribution': risk_contrib
        }
        
        return performance
    
    def efficient_frontier(self, n_points=50, min_weight=0.0, max_weight=1.0):
        """
        Generate efficient frontier
        
        Parameters:
        -----------
        n_points : int
            Number of points on the frontier
        min_weight : float
            Minimum weight constraint
        max_weight : float
            Maximum weight constraint
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with risk and return for each point
        """
        # Find minimum risk and maximum return portfolios
        min_risk_weights = self._min_risk_portfolio(min_weight, max_weight)
        max_return_weights = self._max_return_portfolio(min_weight, max_weight)
        
        min_return = self.portfolio_performance(min_risk_weights)['expected_return']
        max_return = self.portfolio_performance(max_return_weights)['expected_return']
        
        # Generate target returns
        target_returns = np.linspace(min_return, max_return, n_points)
        
        # Initialize results
        frontier_points = []
        
        for target_return in target_returns:
            weights = self._target_return_portfolio(target_return, min_weight, max_weight)
            performance = self.portfolio_performance(weights)
            
            frontier_points.append({
                'return': performance['expected_return'],
                'risk': performance['risk'],
                'sharpe': performance['sharpe_ratio']
            })
        
        # Convert to DataFrame
        frontier = pd.DataFrame(frontier_points)
        
        return frontier
    
    def plot_efficient_frontier(self, n_points=50, show_portfolios=True, figsize=(12, 8), save_path=None):
        """
        Plot efficient frontier
        
        Parameters:
        -----------
        n_points : int
            Number of points on the frontier
        show_portfolios : bool
            Whether to show specific portfolios
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        # Generate efficient frontier
        frontier = self.efficient_frontier(n_points=n_points)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot frontier
        ax.plot(frontier['risk'], frontier['return'], 'b-', linewidth=2)
        
        if show_portfolios:
            # Generate specific portfolios
            portfolios = {
                'Min Risk': self._min_risk_portfolio(0.0, 1.0),
                'Max Sharpe': self._max_sharpe_portfolio(0.0, 1.0),
                'Max Return': self._max_return_portfolio(0.0, 1.0),
                'Risk Parity': self._risk_parity_portfolio(0.0, 1.0),
                'Equal Weight': self._equal_weight_portfolio()
            }
            
            # Calculate performance
            portfolio_performance = {}
            for name, weights in portfolios.items():
                performance = self.portfolio_performance(weights)
                portfolio_performance[name] = {
                    'return': performance['expected_return'],
                    'risk': performance['risk']
                }
            
            # Plot portfolios
            for name, perf in portfolio_performance.items():
                ax.scatter(perf['risk'], perf['return'], s=100, label=name)
                
            # Add legend
            ax.legend()
        
        # Add capital market line if risk-free rate is specified
        if self.risk_free_rate > 0:
            # Find tangency portfolio (max Sharpe ratio)
            max_sharpe_weights = self._max_sharpe_portfolio(0.0, 1.0)
            max_sharpe_perf = self.portfolio_performance(max_sharpe_weights)
            
            # Plot capital market line
            x_range = np.linspace(0, frontier['risk'].max() * 1.2, 100)
            slope = (max_sharpe_perf['expected_return'] - self.risk_free_rate) / max_sharpe_perf['risk']
            y_values = self.risk_free_rate + slope * x_range
            
            ax.plot(x_range, y_values, 'r--', label='Capital Market Line')
            
            # Add risk-free rate point
            ax.scatter(0, self.risk_free_rate, c='r', s=50)
            ax.annotate('Risk-Free Rate', xy=(0, self.risk_free_rate), xytext=(5, 5), 
                       textcoords='offset points')
        
        # Add labels and title
        ax.set_xlabel('Portfolio Risk (Volatility)')
        ax.set_ylabel('Portfolio Expected Return')
        ax.set_title('Efficient Frontier')
        
        # Format axes as percentages
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_risk_contribution(self, weights, figsize=(12, 8), save_path=None):
        """
        Plot risk contribution
        
        Parameters:
        -----------
        weights : pandas.Series
            Portfolio weights
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        # Calculate performance
        performance = self.portfolio_performance(weights)
        
        # Get risk contribution
        risk_contrib = performance['risk_contribution']
        
        # Sort by contribution
        risk_contrib = risk_contrib.sort_values(ascending=False)
        
        # Calculate percentage contribution
        pct_contrib = risk_contrib / risk_contrib.sum() * 100
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot risk contribution
        ax.bar(risk_contrib.index, pct_contrib.values)
        
        # Add labels and title
        ax.set_xlabel('Asset')
        ax.set_ylabel('Risk Contribution (%)')
        ax.set_title('Portfolio Risk Contribution')
        
        # Rotate x-labels
        plt.xticks(rotation=45, ha='right')
        
        # Add grid
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add percentage values on top of bars
        for i, v in enumerate(pct_contrib.values):
            ax.text(i, v + 0.5, f'{v:.1f}%', ha='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_weights(self, weights, figsize=(12, 8), save_path=None):
        """
        Plot portfolio weights
        
        Parameters:
        -----------
        weights : pandas.Series
            Portfolio weights
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        # Sort weights
        sorted_weights = weights.sort_values(ascending=False)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot weights
        ax.bar(sorted_weights.index, sorted_weights.values * 100)
        
        # Add labels and title
        ax.set_xlabel('Asset')
        ax.set_ylabel('Weight (%)')
        ax.set_title('Portfolio Weights')
        
        # Rotate x-labels
        plt.xticks(rotation=45, ha='right')
        
        # Add grid
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add percentage values on top of bars
        for i, v in enumerate(sorted_weights.values):
            ax.text(i, v * 100 + 0.5, f'{v * 100:.1f}%', ha='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def backtest_portfolio(self, weights, start_date=None, end_date=None):
        """
        Backtest portfolio performance
        
        Parameters:
        -----------
        weights : pandas.Series
            Portfolio weights
        start_date : str, optional
            Start date for backtest, if None will use all available data
        end_date : str, optional
            End date for backtest, if None will use all available data
            
        Returns:
        --------
        pandas.Series
            Portfolio returns
        """
        if self.prices is None:
            raise ValueError("Price data is required for backtesting")
            
        # Filter by date if specified
        if start_date is not None or end_date is not None:
            prices = self.prices.copy()
            
            if start_date is not None:
                prices = prices[prices.index >= start_date]
                
            if end_date is not None:
                prices = prices[prices.index <= end_date]
        else:
            prices = self.prices
            
        # Normalize prices
        normalized_prices = prices / prices.iloc[0]
        
        # Calculate portfolio value
        portfolio_values = (normalized_prices * weights).sum(axis=1)
        
        # Calculate returns
        portfolio_returns = portfolio_values.pct_change().dropna()
        
        return portfolio_returns
    
    def plot_backtest(self, weights, benchmark=None, start_date=None, end_date=None, 
                    figsize=(12, 8), save_path=None):
        """
        Plot backtest results
        
        Parameters:
        -----------
        weights : pandas.Series
            Portfolio weights
        benchmark : pandas.Series, optional
            Benchmark returns for comparison
        start_date : str, optional
            Start date for backtest
        end_date : str, optional
            End date for backtest
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        # Run backtest
        portfolio_returns = self.backtest_portfolio(weights, start_date, end_date)
        
        # Calculate cumulative returns
        portfolio_cumulative = (1 + portfolio_returns).cumprod()
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot portfolio returns
        portfolio_cumulative.plot(ax=ax, label='Portfolio')
        
        # Plot benchmark if provided
        if benchmark is not None:
            if start_date is not None:
                benchmark = benchmark[benchmark.index >= start_date]
                
            if end_date is not None:
                benchmark = benchmark[benchmark.index <= end_date]
                
            benchmark_cumulative = (1 + benchmark).cumprod()
            benchmark_cumulative.plot(ax=ax, label='Benchmark')
        
        # Add labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Returns')
        ax.set_title('Portfolio Backtest Results')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(linestyle='--', alpha=0.7)
        
        # Format y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}x'))
        
        # Calculate and display performance metrics
        performance_metrics = self.calculate_performance_metrics(portfolio_returns, benchmark)
        
        # Add metrics as text
        metrics_text = '\n'.join([
            f"Annualized Return: {performance_metrics['annualized_return']:.2%}",
            f"Annualized Volatility: {performance_metrics['annualized_volatility']:.2%}",
            f"Sharpe Ratio: {performance_metrics['sharpe_ratio']:.2f}",
            f"Max Drawdown: {performance_metrics['max_drawdown']:.2%}",
            f"Calmar Ratio: {performance_metrics['calmar_ratio']:.2f}"
        ])
        
        # Position the text box in figure coords
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def calculate_performance_metrics(self, returns, benchmark=None):
        """
        Calculate performance metrics
        
        Parameters:
        -----------
        returns : pandas.Series
            Portfolio returns
        benchmark : pandas.Series, optional
            Benchmark returns for comparison
            
        Returns:
        --------
        dict
            Performance metrics
        """
        # Calculate metrics
        annualized_return = returns.mean() * 252
        annualized_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - self.risk_free_rate) / annualized_volatility
        
        # Calculate drawdown
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative / peak) - 1
        max_drawdown = drawdown.min()
        
        # Calculate Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown)
        
        # Calculate benchmark relative metrics
        if benchmark is not None:
            # Align benchmark with returns
            aligned_benchmark = benchmark.reindex(returns.index)
            
            # Calculate metrics
            excess_returns = returns - aligned_benchmark
            tracking_error = excess_returns.std() * np.sqrt(252)
            information_ratio = excess_returns.mean() * 252 / tracking_error
            
            # Beta
            covariance = returns.cov(aligned_benchmark)
            benchmark_variance = aligned_benchmark.var()
            beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
            
            # Alpha
            benchmark_return = aligned_benchmark.mean() * 252
            alpha = annualized_return - self.risk_free_rate - beta * (benchmark_return - self.risk_free_rate)
        else:
            excess_returns = None
            tracking_error = None
            information_ratio = None
            beta = None
            alpha = None
        
        # Create metrics dictionary
        metrics = {
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'excess_returns': excess_returns,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'beta': beta,
            'alpha': alpha
        }
        
        return metrics
    
    def compare_optimizations(self, methods=None, min_weight=0.0, max_weight=1.0, use_topology=True):
        """
        Compare different optimization methods
        
        Parameters:
        -----------
        methods : list, optional
            List of optimization methods to compare
        min_weight : float
            Minimum weight constraint
        max_weight : float
            Maximum weight constraint
        use_topology : bool
            Whether to use topology analysis for optimization
            
        Returns:
        --------
        pandas.DataFrame
            Comparison of performance metrics
        """
        if methods is None:
            methods = ['min_risk', 'max_sharpe', 'max_return', 'risk_parity', 'equal_weight']
            
        # Initialize results
        results = []
        
        for method in methods:
            # Optimize portfolio
            weights = self.optimize_portfolio(
                method=method, min_weight=min_weight, max_weight=max_weight,
                use_topology=use_topology
            )
            
            # Calculate performance
            performance = self.portfolio_performance(weights)
            
            # Add to results
            results.append({
                'method': method,
                'expected_return': performance['expected_return'],
                'risk': performance['risk'],
                'sharpe_ratio': performance['sharpe_ratio'],
                'diversification_ratio': performance['diversification_ratio']
            })
        
        # Convert to DataFrame
        comparison = pd.DataFrame(results)
        
        # Set method as index
        comparison.set_index('method', inplace=True)
        
        return comparison
    
    def plot_comparison(self, comparison, figsize=(12, 10), save_path=None):
        """
        Plot comparison of optimization methods
        
        Parameters:
        -----------
        comparison : pandas.DataFrame
            Comparison DataFrame from compare_optimizations
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # Plot expected return
        comparison['expected_return'].plot(kind='bar', ax=axes[0])
        axes[0].set_title('Expected Return')
        axes[0].set_ylabel('Annualized Return')
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2%}'))
        
        # Plot risk
        comparison['risk'].plot(kind='bar', ax=axes[1])
        axes[1].set_title('Portfolio Risk')
        axes[1].set_ylabel('Annualized Volatility')
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2%}'))
        
        # Plot Sharpe ratio
        comparison['sharpe_ratio'].plot(kind='bar', ax=axes[2])
        axes[2].set_title('Sharpe Ratio')
        
        # Plot diversification ratio
        comparison['diversification_ratio'].plot(kind='bar', ax=axes[3])
        axes[3].set_title('Diversification Ratio')
        
        # Adjust layout
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig