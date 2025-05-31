#!/usr/bin/env python
"""
Model validation script for quality gates and deployment decisions
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
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

class ModelValidator:
    """Model validation for deployment decisions"""
    
    def __init__(self, 
                 experiment_name: str = "potato-disease-classification",
                 model_name: str = "potato-disease-model",
                 mlflow_uri: str = None):
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.mlflow_uri = mlflow_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        
        # Initialize MLflow
        mlflow.set_tracking_uri(self.mlflow_uri)
        self.client = MlflowClient(tracking_uri=self.mlflow_uri)

    def get_latest_model_run(self):
        """Get the latest model training run"""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                raise ValueError(f"Experiment {self.experiment_name} not found")
            
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="tags.mlflow.runName LIKE 'model_training%'",
                order_by=["start_time DESC"],
                max_results=1
            )
            
            if runs.empty:
                raise ValueError("No training runs found")
            
            return runs.iloc[0]
            
        except Exception as e:
            logger.error(f"Failed to get latest model run: {e}")
            raise

    def get_production_model_metrics(self):
        """Get metrics from current production model"""
        try:
            # Get production model version
            production_versions = self.client.get_latest_versions(
                self.model_name, 
                stages=["Production"]
            )
            
            if not production_versions:
                logger.info("No production model found")
                return None
            
            production_version = production_versions[0]
            run_id = production_version.run_id
            
            # Get run metrics
            run = self.client.get_run(run_id)
            return run.data.metrics
            
        except Exception as e:
            logger.warning(f"Failed to get production model metrics: {e}")
            return None

    def validate_model_quality(self, model_run):
        """Validate model against quality thresholds"""
        quality_thresholds = {
            'accuracy': 0.85,
            'f1_score': 0.80,
            'precision': 0.80,
            'recall': 0.75
        }
        
        validation_results = {}
        overall_pass = True
        
        for metric, threshold in quality_thresholds.items():
            metric_value = model_run.get(f"metrics.{metric}", 0)
            passed = metric_value >= threshold
            
            validation_results[metric] = {
                'value': metric_value,
                'threshold': threshold,
                'passed': passed
            }
            
            if not passed:
                overall_pass = False
        
        validation_results['overall_quality_pass'] = overall_pass
        
        return validation_results

    def compare_with_production(self, new_model_run, production_metrics):
        """Compare new model with production model"""
        if production_metrics is None:
            return {
                'comparison_available': False,
                'improvement_detected': True,  # Deploy if no production model exists
                'message': 'No production model found. New model will be deployed.'
            }
        
        comparison_results = {
            'comparison_available': True,
            'metrics_comparison': {},
            'improvement_detected': False,
            'significant_improvement': False
        }
        
        # Compare key metrics
        key_metrics = ['accuracy', 'f1_score', 'precision', 'recall']
        improvements = 0
        significant_improvements = 0
        
        for metric in key_metrics:
            new_value = new_model_run.get(f"metrics.{metric}", 0)
            old_value = production_metrics.get(metric, 0)
            
            improvement = new_value - old_value
            improvement_percentage = (improvement / old_value * 100) if old_value > 0 else 0
            
            comparison_results['metrics_comparison'][metric] = {
                'new_value': new_value,
                'old_value': old_value,
                'improvement': improvement,
                'improvement_percentage': improvement_percentage,
                'improved': improvement > 0
            }
            
            if improvement > 0:
                improvements += 1
            
            # Significant improvement threshold (2% for most metrics, 1% for accuracy)
            significance_threshold = 0.01 if metric == 'accuracy' else 0.02
            if improvement > significance_threshold:
                significant_improvements += 1
        
        # Determine if improvement is detected
        comparison_results['improvement_detected'] = improvements >= 2  # At least 2 metrics improved
        comparison_results['significant_improvement'] = significant_improvements >= 1  # At least 1 significant improvement
        
        # Overall recommendation
        if comparison_results['significant_improvement']:
            comparison_results['message'] = 'Significant improvement detected. Deployment recommended.'
        elif comparison_results['improvement_detected']:
            comparison_results['message'] = 'Moderate improvement detected. Deployment recommended.'
        else:
            comparison_results['message'] = 'No significant improvement detected. Deployment not recommended.'
        
        return comparison_results

    def validate_model_artifacts(self, model_run):
        """Validate that all required model artifacts exist"""
        artifacts_validation = {
            'required_artifacts': [
                'model',
                'confusion_matrix.png',
                'classification_report',
                'model_metadata'
            ],
            'artifacts_found': [],
            'artifacts_missing': [],
            'validation_passed': True
        }
        
        try:
            run_id = model_run['run_id']
            artifacts = self.client.list_artifacts(run_id)
            
            artifact_names = [artifact.path for artifact in artifacts]
            
            for required_artifact in artifacts_validation['required_artifacts']:
                if any(required_artifact in name for name in artifact_names):
                    artifacts_validation['artifacts_found'].append(required_artifact)
                else:
                    artifacts_validation['artifacts_missing'].append(required_artifact)
                    artifacts_validation['validation_passed'] = False
            
        except Exception as e:
            logger.error(f"Failed to validate artifacts: {e}")
            artifacts_validation['validation_passed'] = False
            artifacts_validation['error'] = str(e)
        
        return artifacts_validation

    def make_deployment_decision(self, quality_validation, comparison_results, artifacts_validation):
        """Make final deployment decision based on all validations"""
        decision = {
            'deploy_recommended': False,
            'stage_recommendation': 'None',
            'reasons': [],
            'blocking_issues': []
        }
        
        # Check quality gates
        if not quality_validation['overall_quality_pass']:
            decision['blocking_issues'].append('Model does not meet quality thresholds')
        
        # Check artifacts
        if not artifacts_validation['validation_passed']:
            decision['blocking_issues'].append('Required model artifacts are missing')
        
        # If no blocking issues, make deployment decision
        if not decision['blocking_issues']:
            if comparison_results['significant_improvement'] or not comparison_results['comparison_available']:
                decision['deploy_recommended'] = True
                decision['stage_recommendation'] = 'Production'
                decision['reasons'].append('Significant improvement over production model')
            elif comparison_results['improvement_detected']:
                decision['deploy_recommended'] = True
                decision['stage_recommendation'] = 'Staging'
                decision['reasons'].append('Moderate improvement detected - deploy to staging first')
            else:
                decision['stage_recommendation'] = 'None'
                decision['reasons'].append('No significant improvement over production model')
        
        return decision

    def execute_deployment(self, model_run, deployment_decision):
        """Execute model deployment based on decision"""
        if not deployment_decision['deploy_recommended']:
            logger.info("Deployment not recommended, skipping...")
            return False
        
        try:
            run_id = model_run['run_id']
            stage = deployment_decision['stage_recommendation']
            
            # Register model if not already registered
            model_uri = f"runs:/{run_id}/model"
            
            try:
                # Create registered model if it doesn't exist
                self.client.create_registered_model(self.model_name)
            except Exception:
                # Model already exists
                pass
            
            # Create model version
            model_version = self.client.create_model_version(
                name=self.model_name,
                source=model_uri,
                run_id=run_id
            )
            
            # Transition to appropriate stage
            if stage in ['Staging', 'Production']:
                self.client.transition_model_version_stage(
                    name=self.model_name,
                    version=model_version.version,
                    stage=stage
                )
                
                logger.info(f"Model version {model_version.version} deployed to {stage}")
            
            return True
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False

    def run_validation(self):
        """Run complete model validation pipeline"""
        logger.info("Starting model validation...")
        
        validation_experiment = "model-validation"
        
        try:
            experiment = mlflow.get_experiment_by_name(validation_experiment)
            if experiment is None:
                experiment_id = mlflow.create_experiment(validation_experiment)
            else:
                experiment_id = experiment.experiment_id
        except Exception:
            experiment_id = "0"  # Default experiment
        
        with mlflow.start_run(experiment_id=experiment_id, run_name="model_validation") as run:
            try:
                # Get latest model run
                latest_model_run = self.get_latest_model_run()
                
                # Get production model metrics
                production_metrics = self.get_production_model_metrics()
                
                # Validate model quality
                quality_validation = self.validate_model_quality(latest_model_run)
                
                # Compare with production
                comparison_results = self.compare_with_production(latest_model_run, production_metrics)
                
                # Validate artifacts
                artifacts_validation = self.validate_model_artifacts(latest_model_run)
                
                # Make deployment decision
                deployment_decision = self.make_deployment_decision(
                    quality_validation, comparison_results, artifacts_validation
                )
                
                # Create validation report
                validation_report = {
                    'timestamp': datetime.now().isoformat(),
                    'model_run_id': latest_model_run['run_id'],
                    'quality_validation': quality_validation,
                    'comparison_results': comparison_results,
                    'artifacts_validation': artifacts_validation,
                    'deployment_decision': deployment_decision
                }
                
                # Log results to MLflow
                mlflow.log_param("validated_run_id", latest_model_run['run_id'])
                mlflow.log_param("quality_gates_passed", quality_validation['overall_quality_pass'])
                mlflow.log_param("deployment_recommended", deployment_decision['deploy_recommended'])
                mlflow.log_param("target_stage", deployment_decision['stage_recommendation'])
                
                # Log metrics
                for metric, result in quality_validation.items():
                    if isinstance(result, dict) and 'value' in result:
                        mlflow.log_metric(f"validated_{metric}", result['value'])
                
                # Save validation report
                mlflow.log_dict(validation_report, "validation_report.json")
                
                # Execute deployment if recommended
                deployment_success = False
                if deployment_decision['deploy_recommended']:
                    deployment_success = self.execute_deployment(latest_model_run, deployment_decision)
                    mlflow.log_param("deployment_executed", deployment_success)
                
                logger.info("Model validation completed")
                logger.info(f"Quality gates passed: {quality_validation['overall_quality_pass']}")
                logger.info(f"Deployment recommended: {deployment_decision['deploy_recommended']}")
                logger.info(f"Target stage: {deployment_decision['stage_recommendation']}")
                
                return deployment_decision['deploy_recommended'] and deployment_success
                
            except Exception as e:
                logger.error(f"Model validation failed: {e}")
                mlflow.log_param("status", "failed")
                mlflow.log_param("error", str(e))
                return False

def main():
    parser = argparse.ArgumentParser(description="Model Validation Pipeline")
    parser.add_argument("--experiment-name", default="potato-disease-classification",
                       help="MLflow experiment name")
    parser.add_argument("--model-name", default="potato-disease-model",
                       help="Model registry name")
    parser.add_argument("--mlflow-uri", help="MLflow tracking URI")
    
    args = parser.parse_args()
    
    # Initialize and run validation
    validator = ModelValidator(
        experiment_name=args.experiment_name,
        model_name=args.model_name,
        mlflow_uri=args.mlflow_uri
    )
    
    success = validator.run_validation()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
