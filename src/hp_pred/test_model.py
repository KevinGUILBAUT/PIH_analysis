from pathlib import Path
import pickle
from itertools import chain, repeat

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.calibration import calibration_curve


class Test_Model:
    def __init__(self, model_configs: list, output_name: str):
        """
        Initialize with model configurations using the new loading logic
        """
        self.model_configs = model_configs
        self.output_name = output_name
        
        # Create figures directory
        self.figures_dir = Path("data/models/figures")
        if not self.figures_dir.exists():
            self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize lists to store components
        self.models = []
        self.test_datasets = []
        self.train_datasets = []
        self.features_names_list = []
        self.model_names = []
        self.model_types = []
        
        # Process each model configuration
        for i, config in enumerate(model_configs):
            # Extract configuration
            test_data = config['test_data']
            train_data = config.get('train_data', None)  # Optional for visualization-only
            model_filename = config.get('model_filename', None)
            features_names = config['features_names']
            model_name = config.get('name', f"model_{i}")
            model_type = config.get('model_type', 'xgboost')
            
            # Load model if filename is provided, otherwise use pre-loaded model
            if model_filename is not None:
                model_path = Path("data/models") / model_filename
                model = self._load_model(model_path, model_type, config)
            else:
                # Use pre-loaded model from config
                model = config['model']
            

            test_data_clean = test_data.dropna(subset=features_names).copy()
            if train_data is not None:
                train_data_clean = train_data.dropna(subset=features_names).copy()
            else:
                train_data_clean = None
                  
            # Clean baseline feature if it exists
            baseline_feature = config.get('baseline_feature', None)
            if baseline_feature is not None:
                test_data_clean = test_data_clean.dropna(subset=[baseline_feature])
                if train_data_clean is not None:
                    train_data_clean = train_data_clean.dropna(subset=[baseline_feature])

            # Store everything
            self.models.append(model)
            self.test_datasets.append(test_data_clean)
            self.train_datasets.append(train_data_clean)
            self.features_names_list.append(features_names)
            self.model_names.append(model_name)
            self.model_types.append(model_type)
            
            print(f"Model {model_name} loaded")
            print(f"Number of points in test data: {len(test_data_clean)}")
            if 'label' in test_data_clean.columns:
                print(f"Prevalence of positive class: {test_data_clean['label'].mean():.2%}")
            print("---")
        
        print(f"Initialized {len(self.models)} models for visualization")

    def _load_model(self, model_path, model_type, config):
        """Load model based on its type"""
        try:
            if model_type == 'xgboost':
                # Load as XGBoost model
                model = xgb.XGBClassifier()
                model.load_model(model_path)
                
            else:
                # Load as pickle model (for other model types)
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                    
            return model
            
        except Exception as e:
            raise ValueError(f"Error loading model {model_path}: {e}")

    def generate_predictions(self):
        """Generate predictions for all models"""
        self.y_pred_models = []
        self.y_true_list = []
        
        for i, (model, test_data, features_names, model_name, model_type) in enumerate(
                zip(self.models, self.test_datasets, self.features_names_list, self.model_names, self.model_types)):
            
            print(f"Generating predictions for {model_name}...")

            if hasattr(model, 'predict_proba'):
                y_pred = model.predict_proba(test_data[features_names])[:, 1]
            else:
                # For models without predict_proba, use predict and assume binary output
                y_pred = model.predict(test_data[features_names])
            
            self.y_pred_models.append(y_pred)
            self.y_true_list.append(test_data["label"].to_numpy())

    def plot_roc_curve(self, model_indices=None, save_fig=True):
        """Plot ROC curves for selected models"""
        if not hasattr(self, "y_pred_models"):
            self.generate_predictions()
            
        if model_indices is None:
            model_indices = list(range(len(self.models)))
            
        plt.figure(figsize=(10, 8))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(model_indices)))
        
        for idx, color in zip(model_indices, colors):
            model_name = self.model_names[idx]
            y_true = self.y_true_list[idx]
            y_pred = self.y_pred_models[idx]
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color=color, lw=2,
                    label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.6, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_fig:
            plt.savefig(self.figures_dir / f"ROC_curves_{self.output_name}.png", 
                       dpi=300, bbox_inches='tight')
        plt.show()

    def plot_precision_recall_curve(self, model_indices=None, save_fig=True):
        """Plot Precision-Recall curves for selected models"""
        if not hasattr(self, "y_pred_models"):
            self.generate_predictions()
            
        if model_indices is None:
            model_indices = list(range(len(self.models)))
            
        plt.figure(figsize=(10, 8))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(model_indices)))
        
        for idx, color in zip(model_indices, colors):
            model_name = self.model_names[idx]
            y_true = self.y_true_list[idx]
            y_pred = self.y_pred_models[idx]
            
            # Calculate PR curve
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            pr_auc = auc(recall, precision)
            
            plt.plot(recall, precision, color=color, lw=2,
                    label=f'{model_name} (AUC = {pr_auc:.3f})')
        
        # Add baseline (random classifier)
        baseline = np.mean([np.mean(y_true) for y_true in self.y_true_list])
        plt.axhline(y=baseline, color='k', linestyle='--', alpha=0.6, 
                   label=f'Baseline (Prevalence = {baseline:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_fig:
            plt.savefig(self.figures_dir / f"PR_curves_{self.output_name}.png", 
                       dpi=300, bbox_inches='tight')
        plt.show()

    def plot_shap_values(self, model_indices=None, feature_groups=None, 
                     max_display=15, save_fig=True):
        """Plot SHAP values for selected models"""
        if model_indices is None:
            model_indices = list(range(len(self.models)))
        
        for idx in model_indices:
            model = self.models[idx]
            model_name = self.model_names[idx]
            test_data = self.test_datasets[idx]
            features_names = self.features_names_list[idx]
            
            print(f"Computing SHAP values for {model_name}...")
            
            # Prepare data
            X_test = test_data[features_names]
            X_train = self.train_datasets[idx][features_names] 
            
            try:
                explainer = shap.Explainer(model, X_train)
                shap_values = explainer(X_test)

                # Bar plot (feature importance)
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_test.columns.tolist(), 
                                plot_type="bar", max_display=max_display, show=False)
                plt.title(f'SHAP Feature Importance - {model_name}')
                if save_fig:
                    plt.savefig(self.figures_dir / f"SHAP_importance_{model_name}_{self.output_name}.png", 
                                dpi=300, bbox_inches='tight')
                plt.show()

                shap.summary_plot(shap_values, X_test, max_display=max_display, show=False)
                plt.gcf().set_size_inches(10, 8)  # fixe la taille de la figure déjà créée
                plt.tight_layout()
                plt.title(f'SHAP Summary Plot - {model_name}', fontsize=14)
                if save_fig:
                    plt.savefig(self.figures_dir / f"SHAP_summary_{model_name}_{self.output_name}.png", 
                                dpi=300, bbox_inches='tight')
                plt.show()

            except Exception as e:
                print(f"Error computing SHAP values for {model_name}: {e}")


    def plot_predicted_probabilities(self, model_indices=None, n_bins=20, save_fig=True):
        """Plot histograms of predicted probabilities vs actual frequency"""
        if not hasattr(self, "y_pred_models"):
            self.generate_predictions()
            
        if model_indices is None:
            model_indices = list(range(len(self.models)))
        
        n_models = len(model_indices)
        fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 10))
        
        if n_models == 1:
            axes = axes.reshape(-1, 1)
        
        for i, idx in enumerate(model_indices):
            model_name = self.model_names[idx]
            y_true = self.y_true_list[idx]
            y_pred = self.y_pred_models[idx]
            
            # Histogram of predicted probabilities
            axes[0, i].hist(y_pred, bins=n_bins, alpha=0.7, density=True)
            axes[0, i].set_title(f'Predicted Probabilities - {model_name}')
            axes[0, i].set_xlabel('Predicted Probability')
            axes[0, i].set_ylabel('Density')
            axes[0, i].grid(True, alpha=0.3)
            
            # Calibration curve 
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_pred, n_bins=n_bins, strategy='uniform'
            )
            
            axes[1, i].plot(mean_predicted_value, fraction_of_positives, 
                           marker='o', linewidth=2, markersize=6)
            axes[1, i].plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Perfect calibration')
            axes[1, i].set_title(f'Calibration Curve - {model_name}')
            axes[1, i].set_xlabel('Mean Predicted Probability')
            axes[1, i].set_ylabel('Fraction of Positives')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_fig:
            plt.savefig(self.figures_dir / f"predicted_probabilities_{self.output_name}.png", 
                       dpi=300, bbox_inches='tight')
        plt.show()

    def run(self, model_indices=None, feature_groups=None, 
            max_shap_display=20, save_figs=True):
        """Run all visualizations"""
        print("Generating predictions...")
        self.generate_predictions()
        
        print("Plotting ROC curves...")
        self.plot_roc_curve(model_indices, save_figs)
        
        print("Plotting Precision-Recall curves...")
        self.plot_precision_recall_curve(model_indices, save_figs)
        
        #print("Plotting predicted probabilities...")
        #self.plot_predicted_probabilities(model_indices, save_fig=save_figs)
        
        print("Computing and plotting SHAP values...")
        self.plot_shap_values(model_indices, feature_groups, max_shap_display, save_figs)
        
