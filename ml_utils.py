def optimize_threshold(y_true, y_pred_proba, beta=2, figsize=(12, 7)):
    """
    Optimize classification threshold using F-beta score and visualize metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred_proba : array-like
        Predicted probabilities for the positive class
    beta : float, default=2
        Beta value for F-beta score (beta > 1 favors recall)
    figsize : tuple, default=(12, 7)
        Figure size for the plot
        
    Returns:
    --------
    dict
        Dictionary containing optimal threshold and corresponding metrics
    """
    from sklearn.metrics import precision_recall_curve
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Calculate precision and recall for different thresholds
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # Calculate F-beta scores
    f_scores = []
    for p, r in zip(precision, recall):
        if p == 0 and r == 0:
            f_scores.append(0)
        else:
            f_score = (1 + beta**2) * (p * r) / ((beta**2 * p) + r)
            f_scores.append(f_score)
    
    # Find optimal threshold
    optimal_idx = np.argmax(f_scores)
    optimal_threshold = pr_thresholds[optimal_idx] if optimal_idx < len(pr_thresholds) else 1.0
    
    # Get optimal metrics
    opt_precision = precision[optimal_idx]
    opt_recall = recall[optimal_idx]
    opt_fscore = f_scores[optimal_idx]
    
    # Plotting
    plt.figure(figsize=figsize)
    
    # Plot metrics
    plt.plot(pr_thresholds, precision[:-1], 'b-', label='Precision')
    plt.plot(pr_thresholds, recall[:-1], 'g-', label='Recall')
    plt.plot(pr_thresholds, f_scores[:-1], 'r-', label=f'F-{beta} Score')
    
    # Add vertical line for optimal threshold
    plt.vlines(optimal_threshold, 0, 1, colors='purple', linestyles='dashed',
              label=f'Optimal threshold: {optimal_threshold:.3f}')
    
    # Plot optimal points
    plt.plot(optimal_threshold, opt_precision, 'b*', markersize=10)
    plt.plot(optimal_threshold, opt_recall, 'g*', markersize=10)
    plt.plot(optimal_threshold, opt_fscore, 'r*', markersize=10)
    
    # Add annotations
    plt.annotate(f'P={opt_precision:.3f}',
                (optimal_threshold, opt_precision),
                xytext=(10, 30), textcoords='offset points')
    plt.annotate(f'F={opt_fscore:.3f}',
                (optimal_threshold, opt_fscore),
                xytext=(10, 40), textcoords='offset points')
    plt.annotate(f'R={opt_recall:.3f}',
                (optimal_threshold, opt_recall),
                xytext=(10, 50), textcoords='offset points')
    
    # Customize plot
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F-Score vs Threshold')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    
    # Print results
    print(f"\nAt optimal threshold {optimal_threshold:.3f}:")
    print(f"Precision: {opt_precision:.3f}")
    print(f"Recall: {opt_recall:.3f}")
    print(f"F-{beta} Score: {opt_fscore:.3f}")
    
    # Return results
    return {
        'threshold': optimal_threshold,
        'precision': opt_precision,
        'recall': opt_recall,
        'f_beta': opt_fscore,
        'beta': beta
    }

# Example usage:
if __name__ == "__main__":
    # Example with TabNet
    y_pred_proba = clf.predict_proba(X_test.values)[:, 1]
    results = optimize_threshold(y_test, y_pred_proba, beta=2)
    
    # Use the optimal threshold
    y_pred = (y_pred_proba >= results['threshold']).astype(int)