"""
Market Structure Analysis Pipeline

This module implements a comprehensive analysis pipeline that combines all the
components for a complete market structure analysis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import other modules
from data_acquisition import get_market_data
from market_topology import MarketTopologyAnalyzer
from regime_detection import MarketRegimeDetector
from gnn_prediction import MarketGNNPredictor
from portfolio_optimization import PortfolioOptimizer


def run_market_structure_analysis(tickers, start_date, end_date, output_dir=None):
    """
    Run complete market structure analysis
    
    Parameters:
    -----------
    tickers : list
        List of ticker symbols
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    output_dir : str, optional
        Directory to save output files
    
    Returns:
    --------
    dict
        Results of the analysis
    """
    print("Starting Market Structure Analysis...")
    print(f"Analyzing {len(tickers)} assets from {start_date} to {end_date}")
    
    # Create output directory if specified
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    # Step 1: Get market data
    print("\nFetching market data from Yahoo Finance...")
    prices, returns = get_market_data(tickers, start_date, end_date)
    
    print(f"Data loaded: {returns.shape[0]} days, {returns.shape[1]} assets")
    
    # Step 2: Analyze market topology
    print("\nAnalyzing market topology using computational geometry...")
    topology = MarketTopologyAnalyzer(returns)
    
    # Compute embedding
    topology.compute_embedding(method='tsne')
    
    # Compute Delaunay triangulation
    topology.compute_delaunay()
    
    # Compute Voronoi diagram
    topology.compute_voronoi()
    
    # Compute minimum spanning tree
    topology.compute_mst()
    
    # Identify communities
    communities = topology.identify_communities()
    
    # Find central assets
    central_assets = topology.identify_central_assets(centrality_measure='betweenness', top_n=5)
    print(f"Top 5 central assets: {central_assets}")
    
    # Find market clusters
    cluster_labels = topology.find_market_clusters()
    
    # Visualize results
    if output_dir:
        print("\nGenerating visualizations...")
        topology.visualize_delaunay(save_path=os.path.join(output_dir, 'delaunay.png'))
        topology.visualize_voronoi(save_path=os.path.join(output_dir, 'voronoi.png'))
        topology.visualize_mst(with_communities=True, save_path=os.path.join(output_dir, 'mst.png'))
        topology.plot_clusters(labels=cluster_labels, save_path=os.path.join(output_dir, 'clusters.png'))
        topology.plot_correlation_network(threshold=0.4, save_path=os.path.join(output_dir, 'correlation_network.png'))
    
    # Step 3: Detect market regimes
    print("\nDetecting market regimes...")
    regime_detector = MarketRegimeDetector(returns, window_size=60, step_size=20)
    
    # Detect regimes
    regime_labels = regime_detector.detect_regimes(n_regimes=4, method='graph', use_hmm=False)
    
    # Get regime periods
    regime_periods = regime_detector.regime_periods
    print(f"Detected {len(regime_periods)} regime periods:")
    for i, period in regime_periods.iterrows():
        print(f"  Regime {period['regime']}: {period['start_date'].date()} to {period['end_date'].date()} "
             f"({period['duration']} days)")
    
    # Visualize regimes
    if output_dir:
        # Create portfolio returns for visualization
        portfolio_returns = returns.mean(axis=1)
        portfolio_cumulative = (1 + portfolio_returns).cumprod()
        
        regime_detector.plot_regime_labels(save_path=os.path.join(output_dir, 'regime_labels.png'))
        regime_detector.plot_regime_transitions(returns_index=portfolio_cumulative, 
                                             save_path=os.path.join(output_dir, 'regime_transitions.png'))
        regime_detector.plot_regime_characteristics(save_path=os.path.join(output_dir, 'regime_characteristics.png'))
    
    # Step 4: Build GNN model for prediction (if enough data)
    if len(returns) > 200:
        print("\nBuilding GNN model for predictive modeling...")
        gnn_predictor = MarketGNNPredictor(returns, prediction_horizon=5, 
                                         lookback_window=60, feature_window=20)
        
        # Prepare data
        train_loader, val_loader, test_loader = gnn_predictor.prepare_data(
            test_ratio=0.2, val_ratio=0.1, correlation_threshold=0.3
        )
        
        # Build model
        model = gnn_predictor.build_model(
            input_dim=15,  # From _extract_asset_features
            hidden_dim=64,
            output_dim=1,
            num_layers=2,
            dropout=0.5,
            gnn_type='gcn',
            pooling='mean'
        )
        
        # Train model
        history = gnn_predictor.train(
            train_loader, val_loader, epochs=100, lr=0.001, weight_decay=5e-4, 
            patience=20, verbose=True
        )
        
        # Evaluate model
        metrics, preds, targets = gnn_predictor.evaluate(test_loader)
        print("\nGNN Model Evaluation Metrics:")
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  MAE: {metrics['mae']:.6f}")
        print(f"  R²: {metrics['r2']:.4f}")
        print(f"  Directional Accuracy: {metrics['dir_acc']:.2%}")
        
        # Visualize results
        if output_dir:
            gnn_predictor.plot_predictions(preds, targets, save_path=os.path.join(output_dir, 'gnn_predictions.png'))
            gnn_predictor.plot_training_history(history, save_path=os.path.join(output_dir, 'gnn_training.png'))
    else:
        print("\nSkipping GNN model (not enough data).")
        gnn_predictor = None
    
    # Step 5: Portfolio optimization
    print("\nOptimizing portfolio using computational geometry insights...")
    optimizer = PortfolioOptimizer(returns, prices, risk_free_rate=0.02)
    
    # Compare different optimization methods
    comparison = optimizer.compare_optimizations(
        methods=['min_risk', 'max_sharpe', 'max_return', 'risk_parity', 'equal_weight'],
        use_topology=True
    )
    
    print("\nOptimization Methods Comparison:")
    print(comparison)
    
    # Get optimal weights using topology information
    weights = optimizer.optimize_portfolio(method='max_sharpe', use_topology=True, topology=topology)
    
    # Performance
    performance = optimizer.portfolio_performance(weights)
    print("\nOptimal Portfolio Performance (Topology-based):")
    print(f"  Expected Return: {performance['expected_return']:.2%}")
    print(f"  Risk: {performance['risk']:.2%}")
    print(f"  Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
    print(f"  Diversification Ratio: {performance['diversification_ratio']:.2f}")
    
    # Visualize results
    if output_dir:
        optimizer.plot_comparison(comparison, save_path=os.path.join(output_dir, 'optimization_comparison.png'))
        optimizer.plot_efficient_frontier(show_portfolios=True, save_path=os.path.join(output_dir, 'efficient_frontier.png'))
        optimizer.plot_weights(weights, save_path=os.path.join(output_dir, 'portfolio_weights.png'))
        optimizer.plot_risk_contribution(weights, save_path=os.path.join(output_dir, 'risk_contribution.png'))
        
        # Backtest portfolio if price data available
        if prices is not None:
            optimizer.plot_backtest(weights, save_path=os.path.join(output_dir, 'portfolio_backtest.png'))
    
    # Collect results
    results = {
        'prices': prices,
        'returns': returns,
        'topology': topology,
        'communities': communities,
        'central_assets': central_assets,
        'cluster_labels': cluster_labels,
        'regime_detector': regime_detector,
        'regime_periods': regime_periods,
        'gnn_predictor': gnn_predictor,
        'optimizer': optimizer,
        'optimal_weights': weights,
        'optimization_comparison': comparison
    }
    
    print("\nMarket Structure Analysis Complete.")
    
    return results