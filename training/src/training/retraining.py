"""
Automated Retraining Orchestrator
Manages model retraining, validation, and rollback based on drift detection.
"""

import os
import json
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import pickle

logger = logging.getLogger(__name__)


class RetrainingOrchestrator:
    """Orchestrates the complete retraining pipeline with validation & rollback."""
    
    def __init__(self, project_root: str = "training"):
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        self.logs_dir = self.project_root / "logs"
        self.outputs_dir = self.project_root / "outputs"
        
        # Backup directories
        self.backup_dir = self.models_dir / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance thresholds
        self.min_accuracy = 0.94
        self.min_auc = 0.99
        self.min_precision = 0.93
        self.max_latency_ms = 200
        self.max_fraud_rate_change = 0.30  # 30% change triggers alert
        
        logger.info(f"Retraining orchestrator initialized at {self.project_root}")
    
    def check_drift(self, baseline_metrics: Dict, current_metrics: Dict) -> Tuple[bool, Dict]:
        """
        Check if model drift detected.
        
        Args:
            baseline_metrics: Metrics from last successful training
            current_metrics: Metrics from production predictions
        
        Returns:
            (drift_detected, drift_report)
        """
        drift_report = {
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {},
            "overall_drift": False,
            "recommendation": "continue_monitoring"
        }
        
        # Check accuracy
        baseline_acc = baseline_metrics.get("accuracy", self.min_accuracy)
        current_acc = current_metrics.get("accuracy", baseline_acc)
        acc_drop = baseline_acc - current_acc
        
        if acc_drop > 0.02:  # 2% drop triggers investigation
            drift_report["checks"]["accuracy"] = {
                "status": "ALERT",
                "baseline": baseline_acc,
                "current": current_acc,
                "change": acc_drop,
                "threshold": 0.02
            }
            drift_report["overall_drift"] = True
        else:
            drift_report["checks"]["accuracy"] = {
                "status": "OK",
                "baseline": baseline_acc,
                "current": current_acc,
                "change": acc_drop
            }
        
        # Check AUC
        baseline_auc = baseline_metrics.get("roc_auc", self.min_auc)
        current_auc = current_metrics.get("roc_auc", baseline_auc)
        auc_drop = baseline_auc - current_auc
        
        if auc_drop > 0.01:  # 1% drop triggers investigation
            drift_report["checks"]["auc"] = {
                "status": "ALERT",
                "baseline": baseline_auc,
                "current": current_auc,
                "change": auc_drop
            }
            drift_report["overall_drift"] = True
        else:
            drift_report["checks"]["auc"] = {
                "status": "OK",
                "baseline": baseline_auc,
                "current": current_auc,
                "change": auc_drop
            }
        
        # Check fraud rate
        baseline_fraud_rate = baseline_metrics.get("fraud_rate", 0.05)
        current_fraud_rate = current_metrics.get("fraud_rate", baseline_fraud_rate)
        fraud_rate_change = abs(current_fraud_rate - baseline_fraud_rate) / baseline_fraud_rate if baseline_fraud_rate > 0 else 0
        
        if fraud_rate_change > self.max_fraud_rate_change:
            drift_report["checks"]["fraud_rate"] = {
                "status": "ALERT",
                "baseline": baseline_fraud_rate,
                "current": current_fraud_rate,
                "change_pct": fraud_rate_change * 100
            }
            drift_report["overall_drift"] = True
        else:
            drift_report["checks"]["fraud_rate"] = {
                "status": "OK",
                "baseline": baseline_fraud_rate,
                "current": current_fraud_rate,
                "change_pct": fraud_rate_change * 100
            }
        
        # Determine recommendation
        if drift_report["overall_drift"]:
            drift_report["recommendation"] = "trigger_retraining"
        
        logger.info(f"Drift check complete: {drift_report['recommendation']}")
        return drift_report["overall_drift"], drift_report
    
    def validate_new_models(self, new_metrics: Dict) -> Tuple[bool, Dict]:
        """
        Validate that new trained models meet performance thresholds.
        
        Args:
            new_metrics: Metrics from newly trained models
        
        Returns:
            (validation_passed, validation_report)
        """
        validation_report = {
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {},
            "passed": True,
            "confidence": "high"
        }
        
        # Check accuracy
        accuracy = new_metrics.get("accuracy", 0)
        if accuracy >= self.min_accuracy:
            validation_report["checks"]["accuracy"] = {
                "status": "PASS",
                "value": accuracy,
                "threshold": self.min_accuracy
            }
        else:
            validation_report["checks"]["accuracy"] = {
                "status": "FAIL",
                "value": accuracy,
                "threshold": self.min_accuracy
            }
            validation_report["passed"] = False
            validation_report["confidence"] = "low"
        
        # Check AUC
        auc = new_metrics.get("roc_auc", 0)
        if auc >= self.min_auc:
            validation_report["checks"]["auc"] = {
                "status": "PASS",
                "value": auc,
                "threshold": self.min_auc
            }
        else:
            validation_report["checks"]["auc"] = {
                "status": "FAIL",
                "value": auc,
                "threshold": self.min_auc
            }
            validation_report["passed"] = False
            validation_report["confidence"] = "low"
        
        # Check precision
        precision = new_metrics.get("precision", 0)
        if precision >= self.min_precision:
            validation_report["checks"]["precision"] = {
                "status": "PASS",
                "value": precision,
                "threshold": self.min_precision
            }
        else:
            validation_report["checks"]["precision"] = {
                "status": "FAIL",
                "value": precision,
                "threshold": self.min_precision
            }
            validation_report["passed"] = False
        
        # Check latency (if available)
        if "latency_ms" in new_metrics:
            latency = new_metrics["latency_ms"]
            if latency <= self.max_latency_ms:
                validation_report["checks"]["latency"] = {
                    "status": "PASS",
                    "value": latency,
                    "threshold": self.max_latency_ms
                }
            else:
                validation_report["checks"]["latency"] = {
                    "status": "FAIL",
                    "value": latency,
                    "threshold": self.max_latency_ms
                }
                validation_report["confidence"] = "medium"
        
        logger.info(f"Model validation: {'PASS' if validation_report['passed'] else 'FAIL'}")
        return validation_report["passed"], validation_report
    
    def backup_current_models(self, backup_name: Optional[str] = None) -> str:
        """
        Backup current production models before retraining.
        
        Args:
            backup_name: Optional custom backup name
        
        Returns:
            Path to backup directory
        """
        if backup_name is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{timestamp}"
        
        backup_path = self.backup_dir / backup_name
        
        # Copy baseline models
        baseline_src = self.models_dir / "baseline"
        baseline_dst = backup_path / "baseline"
        if baseline_src.exists():
            shutil.copytree(baseline_src, baseline_dst, dirs_exist_ok=True)
        
        # Copy comparison
        comparison_src = self.models_dir / "comparison.json"
        if comparison_src.exists():
            shutil.copy(comparison_src, backup_path / "comparison.json")
        
        logger.info(f"Models backed up to {backup_path}")
        return str(backup_path)
    
    def prepare_retraining_data(self) -> Dict:
        """
        Prepare data for retraining (typically last month of production data).
        
        Returns:
            Data prep report
        """
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "data_sources": {},
            "status": "ready",
            "note": "In production, this would fetch the latest production data"
        }
        
        # Check if training data exists
        processed_dir = self.data_dir / "processed"
        if processed_dir.exists():
            X_train = processed_dir / "X_train.npy"
            y_train = processed_dir / "y_train.npy"
            
            if X_train.exists() and y_train.exists():
                report["data_sources"]["train"] = str(X_train)
                report["data_sources"]["labels"] = str(y_train)
        
        logger.info("Data preparation complete")
        return report
    
    def save_retraining_report(self, report: Dict, filename: str = "retraining_report.json") -> str:
        """Save retraining report to file."""
        report_path = self.outputs_dir / filename
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report saved to {report_path}")
        return str(report_path)
    
    def rollback_to_backup(self, backup_name: str) -> bool:
        """
        Rollback to a previous model backup.
        
        Args:
            backup_name: Name of backup to restore
        
        Returns:
            Success status
        """
        backup_path = self.backup_dir / backup_name
        
        if not backup_path.exists():
            logger.error(f"Backup {backup_name} not found")
            return False
        
        # Create new backup of current models first
        current_backup = self.backup_current_models("pre_rollback")
        
        # Restore from backup
        try:
            baseline_src = backup_path / "baseline"
            baseline_dst = self.models_dir / "baseline"
            
            if baseline_src.exists():
                if baseline_dst.exists():
                    shutil.rmtree(baseline_dst)
                shutil.copytree(baseline_src, baseline_dst)
            
            comparison_src = backup_path / "comparison.json"
            comparison_dst = self.models_dir / "comparison.json"
            if comparison_src.exists():
                shutil.copy(comparison_src, comparison_dst)
            
            logger.info(f"Rollback to {backup_name} successful")
            return True
        
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def list_backups(self) -> List[Dict]:
        """List all available model backups."""
        backups = []
        
        if self.backup_dir.exists():
            for backup_path in sorted(self.backup_dir.iterdir(), reverse=True):
                if backup_path.is_dir():
                    backups.append({
                        "name": backup_path.name,
                        "created": datetime.fromtimestamp(backup_path.stat().st_mtime).isoformat(),
                        "path": str(backup_path)
                    })
        
        return backups
    
    def generate_full_report(self, drift_check: Dict, validation: Dict, data_prep: Dict) -> Dict:
        """Generate comprehensive retraining report."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "drift_detection": drift_check,
            "model_validation": validation,
            "data_preparation": data_prep,
            "available_backups": self.list_backups(),
            "summary": {
                "drift_detected": drift_check.get("overall_drift", False),
                "models_valid": validation.get("passed", False),
                "ready_for_deployment": drift_check.get("overall_drift", False) == False and validation.get("passed", False),
                "confidence": validation.get("confidence", "unknown")
            }
        }


def run_retraining_check() -> Dict:
    """
    Main entry point for retraining orchestration.
    Can be called manually or via CI/CD pipeline.
    """
    orchestrator = RetrainingOrchestrator()
    
    # Load baseline metrics
    comparison_file = orchestrator.models_dir / "comparison.json"
    baseline_metrics = {}
    
    if comparison_file.exists():
        with open(comparison_file) as f:
            comparison = json.load(f)
            # Get GNN model metrics (production model)
            baseline_metrics = comparison.get("gnn", {})
    
    # Simulate current production metrics
    # In production, these would come from the /metrics endpoint
    current_metrics = {
        "accuracy": baseline_metrics.get("accuracy", 0.9935),
        "precision": baseline_metrics.get("precision", 0.9892),
        "roc_auc": baseline_metrics.get("roc_auc", 0.9987),
        "fraud_rate": 0.048,  # Simulated
    }
    
    # Check drift
    drift_detected, drift_report = orchestrator.check_drift(baseline_metrics, current_metrics)
    
    # If drift detected, would trigger retraining
    # For now, just prepare the report
    data_prep = orchestrator.prepare_retraining_data()
    
    # Validate models (would use newly trained models)
    validation_passed, validation_report = orchestrator.validate_new_models(current_metrics)
    
    # Generate full report
    full_report = orchestrator.generate_full_report(drift_report, validation_report, data_prep)
    
    # Save report
    report_path = orchestrator.save_retraining_report(full_report)
    
    return full_report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    report = run_retraining_check()
    print(json.dumps(report, indent=2, default=str))
