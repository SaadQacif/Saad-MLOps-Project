#!/usr/bin/env python3
"""
Local Kubernetes deployment script for MLOps pipeline.
Handles complete deployment workflow including container building and Kubernetes deployment.
"""

import os
import sys
import subprocess
import time
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(command, cwd=None, check=True):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            check=check,
            capture_output=True,
            text=True
        )
        if check:
            logger.info(f"✅ Command succeeded: {command}")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Command failed: {command}")
        logger.error(f"Error: {e.stderr}")
        return False, e.stderr


def check_prerequisites():
    """Check if required tools are installed."""
    logger.info("🔍 Checking prerequisites...")
    
    tools = {
        "docker": "docker --version",
        "kubectl": "kubectl version --client",
    }
    
    missing_tools = []
    
    for tool, command in tools.items():
        success, output = run_command(command, check=False)
        if success:
            logger.info(f"✅ {tool} is available")
        else:
            logger.error(f"❌ {tool} is not available")
            missing_tools.append(tool)
    
    if missing_tools:
        logger.error(f"Missing tools: {', '.join(missing_tools)}")
        logger.error("Please install the missing tools before proceeding")
        return False
    
    return True


def check_kubernetes_cluster():
    """Check if Kubernetes cluster is available."""
    logger.info("🔍 Checking Kubernetes cluster...")
    
    # Check if kubectl can connect to cluster
    success, output = run_command("kubectl cluster-info", check=False)
    if not success:
        logger.error("❌ Cannot connect to Kubernetes cluster")
        logger.info("💡 Suggestions:")
        logger.info("  - Start minikube: minikube start")
        logger.info("  - Check Docker Desktop Kubernetes")
        logger.info("  - Verify kubeconfig")
        return False
    
    logger.info("✅ Kubernetes cluster is available")
    return True


def build_and_load_images():
    """Build and load Docker images into Kubernetes."""
    logger.info("🏗️ Building and loading Docker images...")
    
    # Build images using the build script
    success, output = run_command("python scripts/build_containers.py --container all")
    if not success:
        logger.error("❌ Failed to build containers")
        return False
    
    # Check if we're using minikube and load images
    success, output = run_command("minikube status", check=False)
    if success and "Running" in output:
        logger.info("📦 Loading images into minikube...")
        containers = ["frontend", "mlflow", "training"]
        
        for container in containers:
            success, _ = run_command(f"minikube image load potato-disease-{container}:latest")
            if success:
                logger.info(f"✅ Loaded {container} image into minikube")
            else:
                logger.error(f"❌ Failed to load {container} image into minikube")
                return False
    
    return True


def deploy_kubernetes_resources():
    """Deploy Kubernetes resources."""
    logger.info("🚀 Deploying Kubernetes resources...")
    
    # Define deployment order
    manifests = [
        "kubernetes/namespace.yaml",
        "kubernetes/persistent-volumes.yaml",
        "kubernetes/mlflow-deployment.yaml",
        "kubernetes/frontend-deployment.yaml",
        "kubernetes/ml-pipeline-jobs.yaml"
    ]
    
    for manifest in manifests:
        if not os.path.exists(manifest):
            logger.warning(f"⚠️ Manifest not found: {manifest}")
            continue
        
        success, output = run_command(f"kubectl apply -f {manifest}")
        if success:
            logger.info(f"✅ Applied {manifest}")
        else:
            logger.error(f"❌ Failed to apply {manifest}")
            return False
    
    return True


def wait_for_deployments():
    """Wait for deployments to be ready."""
    logger.info("⏳ Waiting for deployments to be ready...")
    
    deployments = [
        ("mlflow-server", "potato-disease-ml"),
        ("frontend", "potato-disease-ml")
    ]
    
    for deployment, namespace in deployments:
        logger.info(f"Waiting for {deployment} in namespace {namespace}...")
        success, output = run_command(
            f"kubectl wait --for=condition=available --timeout=300s deployment/{deployment} -n {namespace}",
            check=False
        )
        
        if success:
            logger.info(f"✅ {deployment} is ready")
        else:
            logger.warning(f"⚠️ {deployment} may not be ready yet")
    
    # Give additional time for services to stabilize
    logger.info("⏳ Allowing services to stabilize...")
    time.sleep(10)


def setup_port_forwarding():
    """Set up port forwarding for local access."""
    logger.info("🌐 Setting up port forwarding...")
    
    port_forwards = [
        ("mlflow-server", "5000:5000", "potato-disease-ml"),
        ("frontend", "8501:8501", "potato-disease-ml")
    ]
    
    processes = []
    
    for service, ports, namespace in port_forwards:
        logger.info(f"Port forwarding {service} ({ports})...")
        
        # Start port forwarding in background
        process = subprocess.Popen(
            ["kubectl", "port-forward", f"service/{service}", ports, "-n", namespace],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        processes.append((service, process))
        
        # Give some time for port forwarding to establish
        time.sleep(2)
    
    logger.info("🎉 Port forwarding established!")
    logger.info("📱 Access URLs:")
    logger.info("  Frontend: http://localhost:8501")
    logger.info("  MLflow: http://localhost:5000")
    
    return processes


def show_deployment_status():
    """Show deployment status."""
    logger.info("📊 Deployment Status:")
    
    # Show pods
    logger.info("Pods:")
    success, output = run_command("kubectl get pods -n potato-disease-ml")
    if success:
        print(output)
    
    # Show services
    logger.info("Services:")
    success, output = run_command("kubectl get services -n potato-disease-ml")
    if success:
        print(output)


def cleanup_deployment():
    """Clean up deployment."""
    logger.info("🧹 Cleaning up deployment...")
    
    success, output = run_command("kubectl delete namespace potato-disease-ml", check=False)
    if success:
        logger.info("✅ Namespace deleted")
    else:
        logger.warning("⚠️ Failed to delete namespace (may not exist)")


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Deploy MLOps pipeline to local Kubernetes")
    parser.add_argument("--build-only", action="store_true", help="Only build containers")
    parser.add_argument("--deploy-only", action="store_true", help="Only deploy (skip building)")
    parser.add_argument("--cleanup", action="store_true", help="Clean up existing deployment")
    parser.add_argument("--no-port-forward", action="store_true", help="Skip port forwarding")
    
    args = parser.parse_args()
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    try:
        if args.cleanup:
            cleanup_deployment()
            return 0
        
        # Check prerequisites
        if not check_prerequisites():
            return 1
        
        if not check_kubernetes_cluster():
            return 1
        
        # Build containers
        if not args.deploy_only:
            if not build_and_load_images():
                return 1
        
        if args.build_only:
            logger.info("✅ Container build completed!")
            return 0
        
        # Deploy to Kubernetes
        if not deploy_kubernetes_resources():
            return 1
        
        # Wait for deployments
        wait_for_deployments()
        
        # Show status
        show_deployment_status()
        
        # Set up port forwarding
        if not args.no_port_forward:
            processes = setup_port_forwarding()
            
            logger.info("🎉 Deployment completed successfully!")
            logger.info("Press Ctrl+C to stop port forwarding and exit...")
            
            try:
                # Keep port forwarding alive
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("🛑 Stopping port forwarding...")
                for service, process in processes:
                    process.terminate()
                    logger.info(f"Stopped port forwarding for {service}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
