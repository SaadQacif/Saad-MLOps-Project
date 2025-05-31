"""
Script to enable MLflow tracing in the workflow
"""

import os
import sys
import argparse
import mlflow
import logging
from configs.mlflow_tracing import enable_mlflow_tracing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to enable MLflow tracing and run a script."""
    parser = argparse.ArgumentParser(description="Run a Python script with MLflow tracing enabled")
    parser.add_argument("script", help="Python script to run with MLflow tracing")
    parser.add_argument("args", nargs="*", help="Arguments to pass to the script")
    
    args = parser.parse_args()
    
    # Enable MLflow tracing
    enable_mlflow_tracing()
    logger.info(f"MLflow tracing enabled. Running script: {args.script}")
    
    # Execute the script
    script_path = args.script
    script_args = args.args
    
    # Add script's directory to sys.path if it's not already there
    script_dir = os.path.dirname(os.path.abspath(script_path))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    
    # Run the script
    with open(script_path, 'r') as f:
        script_code = f.read()
    
    # Set sys.argv to the script and its arguments
    sys.argv = [script_path] + script_args
    
    exec(script_code, {'__name__': '__main__'})

if __name__ == "__main__":
    main()
