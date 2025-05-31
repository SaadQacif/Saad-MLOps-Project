import os
import json
import sys

def check_run_artifacts(run_id, exp_id):
    """Check artifacts for a specific MLflow run"""
    meta_path = os.path.join('./mlruns', exp_id, run_id, 'meta.yaml')
    print(f'Checking run {run_id} in experiment {exp_id}')
    print(f'Meta file exists: {os.path.exists(meta_path)}')
    
    artifacts_path = os.path.join('./mlruns', exp_id, run_id, 'artifacts')
    print(f'Artifacts directory exists: {os.path.exists(artifacts_path)}')
    
    if os.path.exists(artifacts_path):
        print('Artifacts:')
        for item in os.listdir(artifacts_path):
            if os.path.isdir(os.path.join(artifacts_path, item)):
                print(f'- {item}')
                
    metrics_path = os.path.join('./mlruns', exp_id, run_id, 'metrics')
    if os.path.exists(metrics_path):
        print('Metrics:')
        for metric in os.listdir(metrics_path):
            print(f'- {metric}')

if __name__ == "__main__":
    run_id = '342759ed3cf14495830cf438c0eeaa04'  # Most recent run
    exp_id = '1'  # Potato disease classification experiment
    check_run_artifacts(run_id, exp_id)
