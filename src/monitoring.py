"""
API monitoring module for tracking API usage and performance.
"""

import os
import time
import json
import threading
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict
import logging


class APIMonitor:
    """
    A class for monitoring API usage and performance metrics.
    """
    def __init__(self, log_dir: str = None):
        """
        Initialize the API monitor.
        
        Args:
            log_dir: Directory to store logs
        """
        if log_dir is None:
            log_dir = os.path.join(os.getcwd(), "logs")
        
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger("api_monitor")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = os.path.join(log_dir, "api_monitor.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
        
        # Initialize metrics
        self.metrics_file = os.path.join(log_dir, "metrics.json")
        self.metrics = self._load_metrics()
        
        # Threading lock
        self.lock = threading.Lock()
        
        # Start periodic metrics dump
        self._start_metrics_dump()
    
    def _load_metrics(self) -> Dict:
        """Load metrics from file or create new metrics dict."""
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                self.logger.error(f"Error decoding metrics file: {self.metrics_file}")
        
        return {
            "requests": {
                "total": 0,
                "by_endpoint": defaultdict(int),
                "by_status": defaultdict(int),
                "by_hour": defaultdict(int),
            },
            "latency": {
                "total": [],
                "by_endpoint": defaultdict(list),
            },
            "errors": {
                "total": 0,
                "by_endpoint": defaultdict(int),
                "by_type": defaultdict(int),
            },
            "prediction_counts": defaultdict(int)
        }
    
    def _dump_metrics(self):
        """Dump metrics to file."""
        with self.lock:
            # Convert defaultdicts to regular dicts for serialization
            serializable_metrics = json.loads(json.dumps(self.metrics, default=lambda x: dict(x) if isinstance(x, defaultdict) else x))
            
            # Save metrics
            with open(self.metrics_file, "w") as f:
                json.dump(serializable_metrics, f, indent=2)
    
    def _start_metrics_dump(self, interval: int = 300):
        """Start periodic metrics dump."""
        def dump_periodically():
            while True:
                time.sleep(interval)
                self._dump_metrics()
        
        # Start thread
        thread = threading.Thread(target=dump_periodically, daemon=True)
        thread.start()
    
    def log_request(self, endpoint: str, status_code: int, latency: float, request_data: Dict = None):
        """
        Log a request to the API.
        
        Args:
            endpoint: API endpoint
            status_code: HTTP status code
            latency: Request latency in seconds
            request_data: Request data
        """
        with self.lock:
            # Update request metrics
            self.metrics["requests"]["total"] += 1
            self.metrics["requests"]["by_endpoint"][endpoint] += 1
            self.metrics["requests"]["by_status"][str(status_code)] += 1
            
            # Update latency metrics
            self.metrics["latency"]["total"].append(latency)
            self.metrics["latency"]["by_endpoint"][endpoint].append(latency)
            
            # Update hourly metrics
            hour = datetime.now().strftime("%Y-%m-%d-%H")
            self.metrics["requests"]["by_hour"][hour] += 1
            
            # Update error metrics if status code is 4xx or 5xx
            if status_code >= 400:
                self.metrics["errors"]["total"] += 1
                self.metrics["errors"]["by_endpoint"][endpoint] += 1
            
            # Update prediction metrics if this is a prediction request
            if request_data and "prediction" in request_data:
                prediction = request_data["prediction"]
                if isinstance(prediction, list):
                    for p in prediction:
                        self.metrics["prediction_counts"][str(p)] += 1
                else:
                    self.metrics["prediction_counts"][str(prediction)] += 1
        
        # Log to file
        log_message = f"Endpoint: {endpoint}, Status: {status_code}, Latency: {latency:.4f}s"
        self.logger.info(log_message)
    
    def log_error(self, endpoint: str, error_type: str, error_message: str):
        """
        Log an error.
        
        Args:
            endpoint: API endpoint
            error_type: Type of error
            error_message: Error message
        """
        with self.lock:
            self.metrics["errors"]["total"] += 1
            self.metrics["errors"]["by_endpoint"][endpoint] += 1
            self.metrics["errors"]["by_type"][error_type] += 1
        
        # Log to file
        log_message = f"Error in {endpoint}: {error_type} - {error_message}"
        self.logger.error(log_message)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of metrics.
        
        Returns:
            Dictionary of metrics summary
        """
        with self.lock:
            total_requests = self.metrics["requests"]["total"]
            total_errors = self.metrics["errors"]["total"]
            error_rate = total_errors / total_requests if total_requests > 0 else 0
            
            # Calculate average latency
            latencies = self.metrics["latency"]["total"]
            avg_latency = sum(latencies) / len(latencies) if latencies else 0
            
            # Calculate p95 latency
            p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 20 else 0
            
            # Get top endpoints by usage
            top_endpoints = sorted(
                self.metrics["requests"]["by_endpoint"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            # Get recent hourly trends (last 24 hours)
            all_hours = sorted(self.metrics["requests"]["by_hour"].keys())
            recent_hours = all_hours[-24:] if len(all_hours) >= 24 else all_hours
            hourly_trends = {hour: self.metrics["requests"]["by_hour"][hour] for hour in recent_hours}
            
            return {
                "total_requests": total_requests,
                "total_errors": total_errors,
                "error_rate": error_rate,
                "avg_latency": avg_latency,
                "p95_latency": p95_latency,
                "top_endpoints": dict(top_endpoints),
                "hourly_trends": hourly_trends,
                "status_codes": dict(self.metrics["requests"]["by_status"]),
                "top_predictions": dict(sorted(
                    self.metrics["prediction_counts"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5])
            }
