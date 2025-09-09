"""Multi-annotator evaluation framework for hypothesis annotation stability analysis."""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

from .annotate import annotate_texts_with_concepts
from .evaluation import score_hypotheses, compute_hypothesis_separation_scores


class MultiAnnotatorEvaluator:
    """
    Evaluates hypothesis annotations across multiple small LLM annotators.
    
    This class provides functionality to:
    1. Annotate hypotheses using multiple small LLMs
    2. Measure predictive power of annotations
    3. Assess stability across different annotator models
    4. Generate comprehensive analysis reports
    """
    
    def __init__(self, 
                 annotator_models: List[str],
                 cache_name: str = "multi_annotator_eval",
                 n_workers: int = 10):
        """
        Initialize the multi-annotator evaluator.
        
        Args:
            annotator_models: List of model names to use as annotators
            cache_name: Base cache name for storing annotations
            n_workers: Number of parallel workers for annotation
        """
        self.annotator_models = annotator_models
        self.cache_name = cache_name
        self.n_workers = n_workers
        self.annotations = {}  # Will store annotations for each model
        self.evaluation_results = {}  # Will store evaluation metrics
        
    def annotate_with_all_models(self,
                               texts: List[str],
                               hypotheses: List[str],
                               show_progress: bool = True) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Annotate texts with hypotheses using all configured annotator models.
        
        Args:
            texts: List of texts to annotate
            hypotheses: List of hypothesis concepts to check for
            show_progress: Whether to show progress bars
            
        Returns:
            Dictionary mapping model_name -> {hypothesis: annotations_array}
        """
        print(f"Annotating {len(texts)} texts with {len(hypotheses)} hypotheses using {len(self.annotator_models)} models...")
        
        all_annotations = {}
        
        for model in tqdm(self.annotator_models, desc="Annotating with models"):
            print(f"\nAnnotating with {model}...")
            
            # Create model-specific cache name
            model_cache_name = f"{self.cache_name}_{model.replace('/', '_').replace('-', '_')}"
            
            # Annotate texts with hypotheses
            model_annotations = annotate_texts_with_concepts(
                texts=texts,
                concepts=hypotheses,
                model=model,
                cache_name=model_cache_name,
                n_workers=self.n_workers,
                show_progress=show_progress,
                progress_desc=f"Annotating with {model}"
            )
            
            all_annotations[model] = model_annotations
            
        self.annotations = all_annotations
        return all_annotations
    
    def evaluate_predictive_power(self,
                                y_true: np.ndarray,
                                classification: bool = False,
                                corrected_pval_threshold: float = 0.1) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate how predictive the hypothesis annotations are for each annotator model.
        
        Args:
            y_true: True labels/target values
            classification: Whether this is a classification task
            corrected_pval_threshold: P-value threshold for significance after Bonferroni correction
            
        Returns:
            Dictionary mapping model_name -> evaluation metrics
        """
        print("Evaluating predictive power for each annotator model...")
        
        results = {}
        
        for model, annotations in self.annotations.items():
            print(f"\nEvaluating {model}...")
            
            # Use the existing score_hypotheses function
            metrics, hypothesis_df = score_hypotheses(
                hypothesis_annotations=annotations,
                y_true=y_true,
                classification=classification,
                corrected_pval_threshold=corrected_pval_threshold,
                print_summary=False
            )
            
            # Store results
            results[model] = {
                'metrics': metrics,
                'hypothesis_df': hypothesis_df,
                'n_hypotheses': len(hypotheses),
                'n_significant': metrics['Significant'][0],
                'significance_rate': metrics['Significant'][0] / len(hypotheses)
            }
            
        self.evaluation_results = results
        return results
    
    def compute_inter_annotator_agreement(self) -> Dict[str, float]:
        """
        Compute inter-annotator agreement metrics between all pairs of models.
        
        Returns:
            Dictionary mapping (model1, model2) -> agreement metrics
        """
        print("Computing inter-annotator agreement...")
        
        agreement_metrics = {}
        model_pairs = []
        
        # Get all pairs of models
        models = list(self.annotations.keys())
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                model_pairs.append((models[i], models[j]))
        
        for model1, model2 in tqdm(model_pairs, desc="Computing agreement"):
            annotations1 = self.annotations[model1]
            annotations2 = self.annotations[model2]
            
            # Compute agreement for each hypothesis
            hypothesis_agreements = []
            
            for hypothesis in annotations1.keys():
                if hypothesis in annotations2:
                    ann1 = annotations1[hypothesis]
                    ann2 = annotations2[hypothesis]
                    
                    # Skip if either annotation array is empty
                    if len(ann1) == 0 or len(ann2) == 0:
                        continue
                    
                    # Compute various agreement metrics
                    accuracy = accuracy_score(ann1, ann2)
                    correlation = pearsonr(ann1, ann2)[0] if len(set(ann1)) > 1 and len(set(ann2)) > 1 else 0.0
                    spearman_corr = spearmanr(ann1, ann2)[0] if len(set(ann1)) > 1 and len(set(ann2)) > 1 else 0.0
                    
                    hypothesis_agreements.append({
                        'hypothesis': hypothesis,
                        'accuracy': accuracy,
                        'pearson_correlation': correlation,
                        'spearman_correlation': spearman_corr
                    })
            
            # Average across hypotheses
            if hypothesis_agreements:
                avg_accuracy = np.mean([h['accuracy'] for h in hypothesis_agreements])
                avg_pearson = np.mean([h['pearson_correlation'] for h in hypothesis_agreements])
                avg_spearman = np.mean([h['spearman_correlation'] for h in hypothesis_agreements])
                
                agreement_metrics[(model1, model2)] = {
                    'accuracy': avg_accuracy,
                    'pearson_correlation': avg_pearson,
                    'spearman_correlation': avg_spearman,
                    'n_hypotheses': len(hypothesis_agreements),
                    'hypothesis_details': hypothesis_agreements
                }
        
        return agreement_metrics
    
    def analyze_stability(self) -> Dict[str, Any]:
        """
        Analyze the stability of results across different annotator models.
        
        Returns:
            Dictionary containing stability analysis results
        """
        print("Analyzing stability across annotator models...")
        
        if not self.evaluation_results:
            raise ValueError("Must run evaluate_predictive_power() first")
        
        # Extract key metrics for comparison
        stability_metrics = {}
        
        # Collect metrics across models
        r2_scores = []
        significance_rates = []
        n_significant_hypotheses = []
        
        for model, results in self.evaluation_results.items():
            metrics = results['metrics']
            r2_scores.append(metrics.get('r2', 0.0))
            significance_rates.append(results['significance_rate'])
            n_significant_hypotheses.append(results['n_significant'])
        
        # Compute stability statistics
        stability_metrics = {
            'r2_mean': np.mean(r2_scores),
            'r2_std': np.std(r2_scores),
            'r2_cv': np.std(r2_scores) / np.mean(r2_scores) if np.mean(r2_scores) > 0 else 0,
            'significance_rate_mean': np.mean(significance_rates),
            'significance_rate_std': np.std(significance_rates),
            'n_significant_mean': np.mean(n_significant_hypotheses),
            'n_significant_std': np.std(n_significant_hypotheses),
            'model_performance': {
                model: {
                    'r2': results['metrics'].get('r2', 0.0),
                    'significance_rate': results['significance_rate'],
                    'n_significant': results['n_significant']
                }
                for model, results in self.evaluation_results.items()
            }
        }
        
        return stability_metrics
    
    def generate_comparison_report(self) -> pd.DataFrame:
        """
        Generate a comprehensive comparison report across all annotator models.
        
        Returns:
            DataFrame with comparison metrics
        """
        if not self.evaluation_results:
            raise ValueError("Must run evaluate_predictive_power() first")
        
        # Create comparison dataframe
        comparison_data = []
        
        for model, results in self.evaluation_results.items():
            metrics = results['metrics']
            row = {
                'model': model,
                'r2_score': metrics.get('r2', 0.0),
                'auroc': metrics.get('auroc', 0.0),
                'auprc': metrics.get('auprc', 0.0),
                'n_significant_hypotheses': results['n_significant'],
                'significance_rate': results['significance_rate'],
                'total_hypotheses': results['n_hypotheses']
            }
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by R² score (or AUROC for classification)
        if 'auroc' in comparison_df.columns and comparison_df['auroc'].sum() > 0:
            comparison_df = comparison_df.sort_values('auroc', ascending=False)
        else:
            comparison_df = comparison_df.sort_values('r2_score', ascending=False)
        
        return comparison_df
    
    def plot_comparison_metrics(self, save_path: Optional[str] = None) -> None:
        """
        Create visualization comparing metrics across annotator models.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.evaluation_results:
            raise ValueError("Must run evaluate_predictive_power() first")
        
        comparison_df = self.generate_comparison_report()
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Hypothesis Annotation Performance Across Annotator Models', fontsize=16)
        
        # R² Score comparison
        axes[0, 0].bar(range(len(comparison_df)), comparison_df['r2_score'])
        axes[0, 0].set_title('R² Score Comparison')
        axes[0, 0].set_xlabel('Model')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_xticks(range(len(comparison_df)))
        axes[0, 0].set_xticklabels([m.replace('/', '_') for m in comparison_df['model']], rotation=45)
        
        # Significance rate comparison
        axes[0, 1].bar(range(len(comparison_df)), comparison_df['significance_rate'])
        axes[0, 1].set_title('Significance Rate Comparison')
        axes[0, 1].set_xlabel('Model')
        axes[0, 1].set_ylabel('Significance Rate')
        axes[0, 1].set_xticks(range(len(comparison_df)))
        axes[0, 1].set_xticklabels([m.replace('/', '_') for m in comparison_df['model']], rotation=45)
        
        # Number of significant hypotheses
        axes[1, 0].bar(range(len(comparison_df)), comparison_df['n_significant_hypotheses'])
        axes[1, 0].set_title('Number of Significant Hypotheses')
        axes[1, 0].set_xlabel('Model')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_xticks(range(len(comparison_df)))
        axes[1, 0].set_xticklabels([m.replace('/', '_') for m in comparison_df['model']], rotation=45)
        
        # AUROC comparison (if available)
        if 'auroc' in comparison_df.columns and comparison_df['auroc'].sum() > 0:
            axes[1, 1].bar(range(len(comparison_df)), comparison_df['auroc'])
            axes[1, 1].set_title('AUROC Comparison')
            axes[1, 1].set_xlabel('Model')
            axes[1, 1].set_ylabel('AUROC')
        else:
            axes[1, 1].text(0.5, 0.5, 'AUROC not available\n(regression task)', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('AUROC Comparison')
        
        axes[1, 1].set_xticks(range(len(comparison_df)))
        axes[1, 1].set_xticklabels([m.replace('/', '_') for m in comparison_df['model']], rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_inter_annotator_agreement(self, agreement_metrics: Dict[str, float], 
                                     save_path: Optional[str] = None) -> None:
        """
        Create heatmap visualization of inter-annotator agreement.
        
        Args:
            agreement_metrics: Results from compute_inter_annotator_agreement()
            save_path: Optional path to save the plot
        """
        models = list(self.annotations.keys())
        n_models = len(models)
        
        # Create agreement matrices
        accuracy_matrix = np.zeros((n_models, n_models))
        correlation_matrix = np.zeros((n_models, n_models))
        
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i == j:
                    accuracy_matrix[i, j] = 1.0
                    correlation_matrix[i, j] = 1.0
                elif (model1, model2) in agreement_metrics:
                    accuracy_matrix[i, j] = agreement_metrics[(model1, model2)]['accuracy']
                    correlation_matrix[i, j] = agreement_metrics[(model1, model2)]['pearson_correlation']
                elif (model2, model1) in agreement_metrics:
                    accuracy_matrix[i, j] = agreement_metrics[(model2, model1)]['accuracy']
                    correlation_matrix[i, j] = agreement_metrics[(model2, model1)]['pearson_correlation']
        
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Inter-Annotator Agreement Heatmaps', fontsize=16)
        
        # Accuracy heatmap
        sns.heatmap(accuracy_matrix, 
                   xticklabels=[m.replace('/', '_') for m in models],
                   yticklabels=[m.replace('/', '_') for m in models],
                   annot=True, fmt='.3f', cmap='Blues',
                   ax=axes[0])
        axes[0].set_title('Accuracy Agreement')
        
        # Correlation heatmap
        sns.heatmap(correlation_matrix,
                   xticklabels=[m.replace('/', '_') for m in models],
                   yticklabels=[m.replace('/', '_') for m in models],
                   annot=True, fmt='.3f', cmap='Reds',
                   ax=axes[1])
        axes[1].set_title('Pearson Correlation Agreement')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Agreement plot saved to {save_path}")
        
        plt.show()
    
    def print_summary_report(self) -> None:
        """Print a comprehensive summary report of the multi-annotator evaluation."""
        if not self.evaluation_results:
            print("No evaluation results available. Run evaluate_predictive_power() first.")
            return
        
        print("=" * 80)
        print("MULTI-ANNOTATOR HYPOTHESIS EVALUATION SUMMARY")
        print("=" * 80)
        
        # Model performance comparison
        comparison_df = self.generate_comparison_report()
        print("\nMODEL PERFORMANCE COMPARISON:")
        print("-" * 50)
        print(comparison_df.to_string(index=False))
        
        # Stability analysis
        stability = self.analyze_stability()
        print(f"\nSTABILITY ANALYSIS:")
        print("-" * 50)
        print(f"R² Score - Mean: {stability['r2_mean']:.3f}, Std: {stability['r2_std']:.3f}, CV: {stability['r2_cv']:.3f}")
        print(f"Significance Rate - Mean: {stability['significance_rate_mean']:.3f}, Std: {stability['significance_rate_std']:.3f}")
        print(f"Significant Hypotheses - Mean: {stability['n_significant_mean']:.1f}, Std: {stability['n_significant_std']:.1f}")
        
        # Inter-annotator agreement
        agreement_metrics = self.compute_inter_annotator_agreement()
        if agreement_metrics:
            print(f"\nINTER-ANNOTATOR AGREEMENT:")
            print("-" * 50)
            for (model1, model2), metrics in agreement_metrics.items():
                print(f"{model1} vs {model2}:")
                print(f"  Accuracy: {metrics['accuracy']:.3f}")
                print(f"  Pearson Correlation: {metrics['pearson_correlation']:.3f}")
                print(f"  Spearman Correlation: {metrics['spearman_correlation']:.3f}")
                print(f"  Hypotheses compared: {metrics['n_hypotheses']}")
        
        print("\n" + "=" * 80)

