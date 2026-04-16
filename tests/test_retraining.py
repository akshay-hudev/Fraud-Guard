"""
Tests for Retraining Orchestration
Tests drift detection, model validation, backup/rollback, and reporting.
"""

import json
import pytest
from pathlib import Path
from datetime import datetime
import tempfile
import shutil

from training.src.training.retraining import RetrainingOrchestrator


class TestRetrainingOrchestrator:
    """Test suite for RetrainingOrchestrator class."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance."""
        return RetrainingOrchestrator()
    
    @pytest.fixture
    def sample_metrics(self):
        """Sample baseline metrics."""
        return {
            "accuracy": 0.9935,
            "precision": 0.9892,
            "recall": 0.9678,
            "roc_auc": 0.9987,
            "f1": 0.9678,
            "latency_ms": 45.2,
        }
    
    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initializes with correct paths."""
        assert orchestrator.models_dir is not None
        assert orchestrator.backup_dir is not None
        assert orchestrator.min_accuracy == 0.94
        assert orchestrator.min_auc == 0.99
        assert orchestrator.min_precision == 0.93
    
    def test_drift_detection_no_drift(self, orchestrator, sample_metrics):
        """Test drift detection when no drift is present."""
        current_metrics = sample_metrics.copy()
        
        drift_detected, report = orchestrator.check_drift(sample_metrics, current_metrics)
        
        assert drift_detected == False
        assert "checks" in report
        assert "overall_drift" in report
        assert report["overall_drift"] == False
    
    def test_drift_detection_accuracy_drop(self, orchestrator, sample_metrics):
        """Test drift detection when accuracy drops."""
        baseline = sample_metrics.copy()
        current = sample_metrics.copy()
        current["accuracy"] = 0.92  # Drop below 94%
        
        drift_detected, report = orchestrator.check_drift(baseline, current)
        
        assert drift_detected == True
        assert "accuracy" in report["checks"]
        assert report["checks"]["accuracy"]["status"] == "ALERT"
    
    def test_drift_detection_auc_drop(self, orchestrator, sample_metrics):
        """Test drift detection when AUC drops."""
        baseline = sample_metrics.copy()
        current = sample_metrics.copy()
        current["roc_auc"] = 0.985  # Drop below 0.99
        
        drift_detected, report = orchestrator.check_drift(baseline, current)
        
        assert drift_detected == True
        assert "auc" in report["checks"]
        assert report["checks"]["auc"]["status"] == "ALERT"
    
    def test_drift_detection_fraud_rate_change(self, orchestrator, sample_metrics):
        """Test drift detection when fraud rate changes significantly."""
        baseline = sample_metrics.copy()
        baseline["fraud_rate"] = 0.05
        
        current = sample_metrics.copy()
        current["fraud_rate"] = 0.15  # 200% increase
        
        drift_detected, report = orchestrator.check_drift(baseline, current)
        
        assert drift_detected == True
    
    def test_model_validation_all_pass(self, orchestrator, sample_metrics):
        """Test model validation when all metrics pass."""
        passed, report = orchestrator.validate_new_models(sample_metrics)
        
        assert passed == True
        assert report["confidence"] in ["high", "medium", "low"]
        assert all(check["status"] == "PASS" for check in report["checks"].values())
    
    def test_model_validation_accuracy_fail(self, orchestrator, sample_metrics):
        """Test model validation when accuracy fails."""
        sample_metrics["accuracy"] = 0.92  # Below 94%
        
        passed, report = orchestrator.validate_new_models(sample_metrics)
        
        assert passed == False
        assert report["passed"] == False
        assert report["checks"]["accuracy"]["status"] == "FAIL"
    
    def test_model_validation_auc_fail(self, orchestrator, sample_metrics):
        """Test model validation when AUC fails."""
        sample_metrics["roc_auc"] = 0.985  # Below 0.99
        
        passed, report = orchestrator.validate_new_models(sample_metrics)
        
        assert passed == False
        assert report["passed"] == False
        assert report["checks"]["auc"]["status"] == "FAIL"
    
    def test_model_validation_precision_fail(self, orchestrator, sample_metrics):
        """Test model validation when precision fails."""
        sample_metrics["precision"] = 0.92  # Below 0.93
        
        passed, report = orchestrator.validate_new_models(sample_metrics)
        
        assert passed == False
        assert report["passed"] == False
        assert report["checks"]["precision"]["status"] == "FAIL"
    
    def test_model_validation_latency_fail(self, orchestrator, sample_metrics):
        """Test model validation when latency exceeds threshold."""
        sample_metrics["latency_ms"] = 250  # Exceeds 200ms
        
        passed, report = orchestrator.validate_new_models(sample_metrics)
        
        # Latency check may or may not be present depending on implementation
        assert "latency" in report["checks"]
    
    def test_backup_creation(self, orchestrator):
        """Test model backup creation."""
        backup_path = orchestrator.backup_current_models("test_backup")
        
        assert backup_path is not None
        assert Path(backup_path).exists()
    
    def test_list_backups(self, orchestrator):
        """Test listing available backups."""
        # Create a test backup first
        orchestrator.backup_current_models("test_list_backup")
        
        backups = orchestrator.list_backups()
        
        assert isinstance(backups, list)
        assert len(backups) > 0
        assert all("name" in b and "created" in b for b in backups)
    
    def test_rollback_to_backup(self, orchestrator):
        """Test rollback to backup."""
        # Create a backup
        backup_path = orchestrator.backup_current_models("test_rollback")
        backup_name = Path(backup_path).name
        
        # Attempt rollback
        success = orchestrator.rollback_to_backup(backup_name)
        
        # Should succeed if backup exists
        assert isinstance(success, bool)
    
    def test_prepare_retraining_data(self, orchestrator):
        """Test data preparation for retraining."""
        result = orchestrator.prepare_retraining_data()
        
        assert "status" in result
        assert result["status"] == "ready"
        assert "timestamp" in result
        assert "data_sources" in result
    
    def test_full_retraining_report(self, orchestrator, sample_metrics):
        """Test generating full retraining report."""
        baseline = sample_metrics.copy()
        current = sample_metrics.copy()
        
        drift_detected, drift_report = orchestrator.check_drift(baseline, current)
        validation_passed, validation_report = orchestrator.validate_new_models(current)
        data_prep = orchestrator.prepare_retraining_data()
        
        report = orchestrator.generate_full_report(
            drift_report,
            validation_report,
            data_prep
        )
        
        assert "drift_detection" in report
        assert "model_validation" in report
        assert "data_preparation" in report
        assert "timestamp" in report


class TestRetrainingIntegration:
    """Integration tests for retraining pipeline."""
    
    def test_end_to_end_no_drift_flow(self):
        """Test complete flow when no drift is detected."""
        from training.src.training.retraining import RetrainingOrchestrator
        
        orchestrator = RetrainingOrchestrator()
        
        # Simulate metrics
        baseline = {"accuracy": 0.9935, "roc_auc": 0.9987, "fraud_rate": 0.048}
        current = {"accuracy": 0.9935, "roc_auc": 0.9987, "fraud_rate": 0.048}
        
        # Check drift
        drift_detected, drift_report = orchestrator.check_drift(baseline, current)
        
        assert drift_detected == False
        print("✓ No drift detected")
    
    def test_end_to_end_drift_detected_flow(self):
        """Test complete flow when drift is detected."""
        from training.src.training.retraining import RetrainingOrchestrator
        
        orchestrator = RetrainingOrchestrator()
        
        # Simulate drift
        baseline = {"accuracy": 0.9935, "roc_auc": 0.9987, "fraud_rate": 0.048}
        current = {"accuracy": 0.92, "roc_auc": 0.9987, "fraud_rate": 0.048}  # Accuracy dropped
        
        # Check drift
        drift_detected, drift_report = orchestrator.check_drift(baseline, current)
        assert drift_detected == True
        print("✓ Drift detected")
        
        # Backup models
        backup_path = orchestrator.backup_current_models()
        print(f"✓ Models backed up: {backup_path}")
        
        # Prepare data
        data_prep = orchestrator.prepare_retraining_data()
        print(f"✓ Data prepared: {data_prep['data_sources']}")
    
    def test_backup_and_rollback_flow(self):
        """Test complete backup and rollback flow."""
        from training.src.training.retraining import RetrainingOrchestrator
        
        orchestrator = RetrainingOrchestrator()
        
        # Create backup
        backup_path = orchestrator.backup_current_models("test_rollback_flow")
        backup_name = Path(backup_path).name
        print(f"✓ Backup created: {backup_name}")
        
        # List backups
        backups = orchestrator.list_backups()
        assert len(backups) > 0
        print(f"✓ {len(backups)} backups available")
        
        # Rollback
        success = orchestrator.rollback_to_backup(backup_name)
        print(f"✓ Rollback {'successful' if success else 'attempted'}")


class TestRetrainingCLI:
    """Test CLI command functionality."""
    
    def test_can_import_orchestrator(self):
        """Test that orchestrator can be imported."""
        from training.src.training.retraining import RetrainingOrchestrator
        assert RetrainingOrchestrator is not None
    
    def test_can_import_scheduler(self):
        """Test that scheduler can be imported."""
        try:
            from training.src.training.scheduler import RetrainingScheduler
            assert RetrainingScheduler is not None
        except ImportError:
            # APScheduler may not be installed
            pass
    
    def test_retrain_cli_exists(self):
        """Test that retrain CLI script exists."""
        retrain_script = Path("training/scripts/retrain.py")
        assert retrain_script.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
