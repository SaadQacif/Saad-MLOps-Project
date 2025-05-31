#!/usr/bin/env python
"""
Data and model drift detection for monitoring ML model performance
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import mlflow
from mlflow.tracking import MlflowClient

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DriftDetector:
    """Data and model drift detection system"""
    
    def __init__(self, 
                 reference_data_path: str,
                 current_data_path: str,
                 output_path: str,
                 mlflow_uri: str = None):
        self.reference_data_path = Path(reference_data_path)
        self.current_data_path = Path(current_data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.mlflow_uri = mlflow_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        
        # Initialize MLflow
        mlflow.set_tracking_uri(self.mlflow_uri)
        self.client = MlflowClient(tracking_uri=self.mlflow_uri)

    def load_data(self, data_path: Path):
        """Load data from CSV or directory structure"""
        try:
            if data_path.is_file() and data_path.suffix == '.csv':
                return pd.read_csv(data_path)
            elif data_path.is_dir():
                # Look for recent CSV files in the directory
                csv_files = list(data_path.glob('*.csv'))
                if csv_files:
                    # Use the most recent file
                    latest_file = max(csv_files, key=os.path.getctime)
                    return pd.read_csv(latest_file)
                else:
                    raise FileNotFoundError(f"No CSV files found in {data_path}")
            else:
                raise ValueError(f"Invalid data path: {data_path}")
        except Exception as e:
            logger.error(f"Failed to load data from {data_path}: {e}")
            raise

    def calculate_statistical_drift(self, reference_data, current_data):
        """Calculate statistical drift using various methods"""
        from scipy import stats
        
        drift_results = {}
        
        # Ensure we have the same columns
        common_columns = set(reference_data.columns) & set(current_data.columns)
        numeric_columns = []
        
        for col in common_columns:
            if col != 'class' and pd.api.types.is_numeric_dtype(reference_data[col]):
                numeric_columns.append(col)
        
        logger.info(f"Analyzing drift for {len(numeric_columns)} numeric features")
        
        for column in numeric_columns:
            ref_values = reference_data[col].dropna()
            cur_values = current_data[col].dropna()
            
            # Kolmogorov-Smirnov test
            ks_statistic, ks_p_value = stats.ks_2samp(ref_values, cur_values)
            
            # Mann-Whitney U test
            try:
                mw_statistic, mw_p_value = stats.mannwhitneyu(ref_values, cur_values, alternative='two-sided')
            except Exception:
                mw_statistic, mw_p_value = None, None
            
            # Population Stability Index (PSI)
            psi = self.calculate_psi(ref_values, cur_values)
            
            # Statistical measures
            ref_mean, cur_mean = ref_values.mean(), cur_values.mean()
            ref_std, cur_std = ref_values.std(), cur_values.std()
            
            mean_shift = abs(cur_mean - ref_mean) / ref_std if ref_std > 0 else 0
            std_ratio = cur_std / ref_std if ref_std > 0 else 1
            
            drift_results[column] = {
                'ks_statistic': ks_statistic,
                'ks_p_value': ks_p_value,
                'ks_drift_detected': ks_p_value < 0.05 if ks_p_value is not None else False,
                'mw_statistic': mw_statistic,
                'mw_p_value': mw_p_value,
                'mw_drift_detected': mw_p_value < 0.05 if mw_p_value is not None else False,
                'psi': psi,
                'psi_drift_detected': psi > 0.2,  # PSI > 0.2 indicates significant drift
                'ref_mean': ref_mean,
                'cur_mean': cur_mean,
                'ref_std': ref_std,
                'cur_std': cur_std,
                'mean_shift': mean_shift,
                'std_ratio': std_ratio,
                'mean_drift_detected': mean_shift > 2,  # Mean shifted by more than 2 std devs
                'std_drift_detected': std_ratio > 1.5 or std_ratio < 0.67  # Std changed by more than 50%
            }
        
        return drift_results

    def calculate_psi(self, reference, current, num_bins=10):
        """Calculate Population Stability Index (PSI)"""
        try:
            # Create bins based on reference data
            _, bin_edges = np.histogram(reference, bins=num_bins)
            
            # Make sure the bins cover the range of both datasets
            min_val = min(reference.min(), current.min())
            max_val = max(reference.max(), current.max())
            
            if bin_edges[0] > min_val:
                bin_edges[0] = min_val
            if bin_edges[-1] < max_val:
                bin_edges[-1] = max_val
            
            # Calculate frequencies
            ref_freq, _ = np.histogram(reference, bins=bin_edges)
            cur_freq, _ = np.histogram(current, bins=bin_edges)
            
            # Convert to proportions
            ref_prop = ref_freq / len(reference)
            cur_prop = cur_freq / len(current)
            
            # Avoid division by zero
            ref_prop = np.where(ref_prop == 0, 0.0001, ref_prop)
            cur_prop = np.where(cur_prop == 0, 0.0001, cur_prop)
            
            # Calculate PSI
            psi = np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop))
            
            return psi
        except Exception as e:
            logger.warning(f"Failed to calculate PSI: {e}")
            return 0

    def detect_class_distribution_drift(self, reference_data, current_data):
        """Detect drift in class distribution"""
        class_drift = {}
        
        if 'class' in reference_data.columns and 'class' in current_data.columns:
            ref_class_dist = reference_data['class'].value_counts(normalize=True)
            cur_class_dist = current_data['class'].value_counts(normalize=True)
            
            # Ensure same classes
            all_classes = set(ref_class_dist.index) | set(cur_class_dist.index)
            
            for cls in all_classes:
                ref_prop = ref_class_dist.get(cls, 0)
                cur_prop = cur_class_dist.get(cls, 0)
                
                class_drift[cls] = {
                    'reference_proportion': ref_prop,
                    'current_proportion': cur_prop,
                    'absolute_change': abs(cur_prop - ref_prop),
                    'relative_change': (cur_prop - ref_prop) / ref_prop if ref_prop > 0 else float('inf')
                }
            
            # Chi-square test for overall distribution change
            from scipy.stats import chisquare
            
            # Align distributions
            ref_counts = []
            cur_counts = []
            
            for cls in all_classes:
                ref_counts.append(ref_class_dist.get(cls, 0) * len(reference_data))
                cur_counts.append(cur_class_dist.get(cls, 0) * len(current_data))
            
            try:
                chi2_stat, chi2_p = chisquare(cur_counts, ref_counts)
                class_drift['overall'] = {
                    'chi2_statistic': chi2_stat,
                    'chi2_p_value': chi2_p,
                    'distribution_drift_detected': chi2_p < 0.05
                }
            except Exception as e:
                logger.warning(f"Failed to perform chi-square test: {e}")
                class_drift['overall'] = {
                    'chi2_statistic': None,
                    'chi2_p_value': None,
                    'distribution_drift_detected': False
                }
        
        return class_drift

    def generate_drift_report(self, statistical_drift, class_drift):
        """Generate comprehensive drift report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_features_analyzed': len(statistical_drift),
                'features_with_drift': 0,
                'drift_detection_methods': ['ks_test', 'mannwhitney_u', 'psi', 'mean_shift', 'std_change'],
                'class_distribution_drift': class_drift.get('overall', {}).get('distribution_drift_detected', False)
            },
            'feature_drift': statistical_drift,
            'class_drift': class_drift,
            'recommendations': []
        }
        
        # Count features with drift
        for feature, results in statistical_drift.items():
            drift_detected = any([
                results.get('ks_drift_detected', False),
                results.get('mw_drift_detected', False),
                results.get('psi_drift_detected', False),
                results.get('mean_drift_detected', False),
                results.get('std_drift_detected', False)
            ])
            
            if drift_detected:
                report['summary']['features_with_drift'] += 1
        
        # Generate recommendations
        if report['summary']['features_with_drift'] > 0:
            report['recommendations'].append("Feature drift detected. Consider retraining the model.")
        
        if report['summary']['class_distribution_drift']:
            report['recommendations'].append("Class distribution drift detected. Review data pipeline and consider rebalancing.")
        
        drift_ratio = report['summary']['features_with_drift'] / max(report['summary']['total_features_analyzed'], 1)
        
        if drift_ratio > 0.3:
            report['recommendations'].append("High proportion of features showing drift. Immediate model retraining recommended.")
        elif drift_ratio > 0.1:
            report['recommendations'].append("Moderate feature drift detected. Schedule model retraining.")
        else:
            report['recommendations'].append("Minimal drift detected. Continue monitoring.")
        
        return report

    def run_drift_detection(self):
        """Run complete drift detection pipeline"""
        logger.info("Starting drift detection...")
        
        experiment_name = "drift-detection"
        
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
        except Exception:
            experiment_id = "0"  # Default experiment
        
        with mlflow.start_run(experiment_id=experiment_id, run_name="drift_analysis") as run:
            try:
                # Load data
                logger.info("Loading reference and current data...")
                reference_data = self.load_data(self.reference_data_path)
                current_data = self.load_data(self.current_data_path)
                
                # Log data info
                mlflow.log_param("reference_data_path", str(self.reference_data_path))
                mlflow.log_param("current_data_path", str(self.current_data_path))
                mlflow.log_metric("reference_data_size", len(reference_data))
                mlflow.log_metric("current_data_size", len(current_data))
                
                # Detect statistical drift
                logger.info("Calculating statistical drift...")
                statistical_drift = self.calculate_statistical_drift(reference_data, current_data)
                
                # Detect class distribution drift
                logger.info("Analyzing class distribution drift...")
                class_drift = self.detect_class_distribution_drift(reference_data, current_data)
                
                # Generate report
                report = self.generate_drift_report(statistical_drift, class_drift)
                
                # Log metrics to MLflow
                mlflow.log_metric("features_with_drift", report['summary']['features_with_drift'])
                mlflow.log_metric("total_features_analyzed", report['summary']['total_features_analyzed'])
                mlflow.log_metric("drift_ratio", 
                                 report['summary']['features_with_drift'] / 
                                 max(report['summary']['total_features_analyzed'], 1))
                
                if 'overall' in class_drift:
                    mlflow.log_metric("class_distribution_drift", 
                                     1 if class_drift['overall'].get('distribution_drift_detected', False) else 0)
                
                # Save report
                report_file = self.output_path / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(report_file, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                
                # Log report as artifact
                mlflow.log_artifact(str(report_file))
                
                logger.info(f"Drift detection completed. Report saved to {report_file}")
                logger.info(f"Features with drift: {report['summary']['features_with_drift']}/{report['summary']['total_features_analyzed']}")
                
                return True
                
            except Exception as e:
                logger.error(f"Drift detection failed: {e}")
                mlflow.log_param("status", "failed")
                mlflow.log_param("error", str(e))
                return False

def main():
    parser = argparse.ArgumentParser(description="Data and Model Drift Detection")
    parser.add_argument("--reference-data-path", required=True, 
                       help="Path to reference data (training data)")
    parser.add_argument("--current-data-path", required=True,
                       help="Path to current production data")
    parser.add_argument("--output-path", required=True,
                       help="Path to save drift reports")
    parser.add_argument("--mlflow-uri", help="MLflow tracking URI")
    
    args = parser.parse_args()
    
    # Initialize and run drift detection
    detector = DriftDetector(
        reference_data_path=args.reference_data_path,
        current_data_path=args.current_data_path,
        output_path=args.output_path,
        mlflow_uri=args.mlflow_uri
    )
    
    success = detector.run_drift_detection()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
