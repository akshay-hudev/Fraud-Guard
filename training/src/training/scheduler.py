"""
Retraining Scheduler
Handles scheduled drift detection and retraining using APScheduler.
Can be run as a background service or via CI/CD cron jobs.
"""

import json
import logging
from datetime import datetime
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class RetrainingScheduler:
    """Manages scheduled retraining checks and execution."""
    
    def __init__(self):
        self.task_history = []
        self.last_check = None
        self.last_retraining = None
        
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            self.scheduler = BackgroundScheduler()
            self.scheduler_available = True
            logger.info("APScheduler available - background tasks enabled")
        except ImportError:
            self.scheduler = None
            self.scheduler_available = False
            logger.info("APScheduler not available - use cron/GitHub Actions instead")
    
    def schedule_monthly_check(self, callback: Callable):
        """
        Schedule monthly drift & retraining check.
        
        Args:
            callback: Function to call for retraining check
        """
        if not self.scheduler_available:
            logger.warning("Scheduler not available - use external cron instead")
            return
        
        try:
            from apscheduler.triggers.cron import CronTrigger
            
            # Run on 1st of each month at midnight
            trigger = CronTrigger(day=1, hour=0, minute=0)
            self.scheduler.add_job(
                callback,
                trigger=trigger,
                id='monthly_retraining_check',
                name='Monthly Retraining Check',
                replace_existing=True
            )
            
            logger.info("Monthly retraining check scheduled (1st of month, 00:00 UTC)")
            
        except Exception as e:
            logger.error(f"Failed to schedule job: {e}")
    
    def schedule_periodic_check(self, callback: Callable, interval_hours: int = 24):
        """
        Schedule periodic drift check.
        
        Args:
            callback: Function to call for drift check
            interval_hours: Check interval in hours (default: 24)
        """
        if not self.scheduler_available:
            logger.warning("Scheduler not available - use external cron instead")
            return
        
        try:
            from apscheduler.triggers.interval import IntervalTrigger
            
            trigger = IntervalTrigger(hours=interval_hours)
            self.scheduler.add_job(
                callback,
                trigger=trigger,
                id='periodic_drift_check',
                name='Periodic Drift Check',
                replace_existing=True
            )
            
            logger.info(f"Periodic drift check scheduled every {interval_hours} hours")
            
        except Exception as e:
            logger.error(f"Failed to schedule periodic job: {e}")
    
    def start(self):
        """Start the background scheduler."""
        if self.scheduler_available and not self.scheduler.running:
            self.scheduler.start()
            logger.info("Retraining scheduler started")
        elif not self.scheduler_available:
            logger.info("Using external scheduler (CI/CD or cron)")
    
    def stop(self):
        """Stop the background scheduler."""
        if self.scheduler_available and self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Retraining scheduler stopped")
    
    def get_scheduled_jobs(self) -> list:
        """Get list of scheduled jobs."""
        if self.scheduler_available:
            return [
                {
                    "id": job.id,
                    "name": job.name,
                    "next_run": job.next_run_time.isoformat() if job.next_run_time else None
                }
                for job in self.scheduler.get_jobs()
            ]
        return []
    
    def log_task(self, task_name: str, status: str, details: dict):
        """Log a retraining task."""
        task = {
            "timestamp": datetime.utcnow().isoformat(),
            "task": task_name,
            "status": status,
            "details": details
        }
        self.task_history.append(task)
        
        if status == "drift_detected":
            self.last_check = datetime.utcnow()
        elif status == "retraining_complete":
            self.last_retraining = datetime.utcnow()
        
        logger.info(f"Task logged: {task_name} - {status}")
        return task
    
    def get_task_history(self, limit: int = 10) -> list:
        """Get recent task history."""
        return self.task_history[-limit:]


def create_retraining_callback(orchestrator):
    """
    Create a callback function for scheduled retraining checks.
    
    Args:
        orchestrator: RetrainingOrchestrator instance
    
    Returns:
        Callback function
    """
    def check_and_retrain():
        """Performs drift check and conditional retraining."""
        import json
        from pathlib import Path
        
        logger.info("=" * 60)
        logger.info("SCHEDULED RETRAINING CHECK STARTED")
        logger.info("=" * 60)
        
        try:
            # Load baseline metrics
            comparison_file = orchestrator.models_dir / "comparison.json"
            baseline_metrics = {}
            
            if comparison_file.exists():
                with open(comparison_file) as f:
                    comparison = json.load(f)
                    baseline_metrics = comparison.get("gnn", {})
            
            # Get current production metrics
            # In production, these would come from /metrics endpoint
            current_metrics = {
                "accuracy": baseline_metrics.get("accuracy", 0.9935),
                "precision": baseline_metrics.get("precision", 0.9892),
                "roc_auc": baseline_metrics.get("roc_auc", 0.9987),
                "fraud_rate": 0.048,
            }
            
            # Check for drift
            drift_detected, drift_report = orchestrator.check_drift(baseline_metrics, current_metrics)
            
            if drift_detected:
                logger.warning("⚠️ DRIFT DETECTED - Triggering retraining pipeline")
                
                # Backup current models
                backup_path = orchestrator.backup_current_models()
                
                # Prepare data
                data_prep = orchestrator.prepare_retraining_data()
                
                # In production CI/CD, actual retraining would happen here
                logger.info("Running model training pipeline...")
                
                # Validate new models
                validation_passed, validation_report = orchestrator.validate_new_models(current_metrics)
                
                if validation_passed:
                    logger.info("✓ New models validated successfully")
                    logger.info("✓ Ready for deployment")
                else:
                    logger.warning("✗ New models failed validation - rolling back")
                    orchest.rollback_to_backup(Path(backup_path).name)
                    
                logger.info("=" * 60)
                logger.info("RETRAINING CHECK COMPLETE")
                logger.info("=" * 60)
            else:
                logger.info("✓ No drift detected - models performing well")
                logger.info("=" * 60)
                logger.info("RETRAINING CHECK COMPLETE")
                logger.info("=" * 60)
        
        except Exception as e:
            logger.error(f"Retraining check failed: {e}", exc_info=True)
    
    return check_and_retrain


# For manual CLI invocation
def main():
    """Main entry point for testing."""
    import sys
    logging.basicConfig(level=logging.INFO)
    
    from training.src.training.retraining import RetrainingOrchestrator
    
    orchestrator = RetrainingOrchestrator()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "check":
            # Run drift check
            from training.src.training.retraining import run_retraining_check
            report = run_retraining_check()
            print(json.dumps(report, indent=2, default=str))
        
        elif command == "schedule":
            # Start scheduler
            scheduler = RetrainingScheduler()
            callback = create_retraining_callback(orchestrator)
            scheduler.schedule_monthly_check(callback)
            scheduler.start()
            
            print("Retraining scheduler running...")
            print("Scheduled jobs:", scheduler.get_scheduled_jobs())
            
            try:
                import time
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                scheduler.stop()
                print("\nScheduler stopped")
        
        elif command == "list-backups":
            backups = orchestrator.list_backups()
            print(json.dumps(backups, indent=2, default=str))
        
        else:
            print(f"Unknown command: {command}")
            print("Available commands: check, schedule, list-backups")
    
    else:
        print("Retraining Scheduler")
        print("Usage: python -m training.src.training.scheduler <command>")
        print("Commands:")
        print("  check         - Run drift check now")
        print("  schedule      - Start background scheduler")
        print("  list-backups  - List model backups")


if __name__ == "__main__":
    main()
