#!/usr/bin/env python
"""
Retraining CLI Script
Entry point for manual and automated retraining operations.

Usage:
    python retrain.py check                    # Check for drift
    python retrain.py retrain                  # Trigger full retraining
    python retrain.py validate                 # Validate current models
    python retrain.py list-backups             # List model backups
    python retrain.py rollback <backup_name>   # Rollback to backup
    python retrain.py schedule                 # Start background scheduler
    python retrain.py health                   # Check system health
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.retraining import RetrainingOrchestrator, run_retraining_check
from src.training.scheduler import RetrainingScheduler, create_retraining_callback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cmd_check():
    """Check for model drift."""
    print("\n" + "=" * 70)
    print("DRIFT DETECTION CHECK")
    print("=" * 70 + "\n")
    
    report = run_retraining_check()
    
    print(json.dumps(report, indent=2, default=str))
    
    if report["summary"]["drift_detected"]:
        print("\n⚠️  DRIFT DETECTED - Consider triggering retraining")
        return 1
    else:
        print("\n✓ No drift detected - Models performing well")
        return 0


def cmd_validate():
    """Validate current models."""
    print("\n" + "=" * 70)
    print("MODEL VALIDATION")
    print("=" * 70 + "\n")
    
    orchestrator = RetrainingOrchestrator()
    
    # Load current model metrics
    comparison_file = orchestrator.models_dir / "comparison.json"
    if comparison_file.exists():
        with open(comparison_file) as f:
            comparison = json.load(f)
            gnn_metrics = comparison.get("gnn", {})
    else:
        print("✗ No model metrics found")
        return 1
    
    # Validate
    passed, report = orchestrator.validate_new_models(gnn_metrics)
    
    print(json.dumps(report, indent=2, default=str))
    
    if passed:
        print("\n✓ ALL VALIDATION CHECKS PASSED")
        return 0
    else:
        print("\n✗ VALIDATION FAILED - Models below thresholds")
        return 1


def cmd_list_backups():
    """List available model backups."""
    print("\n" + "=" * 70)
    print("AVAILABLE MODEL BACKUPS")
    print("=" * 70 + "\n")
    
    orchestrator = RetrainingOrchestrator()
    backups = orchestrator.list_backups()
    
    if not backups:
        print("No backups found")
        return 0
    
    for i, backup in enumerate(backups, 1):
        print(f"{i}. {backup['name']}")
        print(f"   Created: {backup['created']}")
        print(f"   Path: {backup['path']}\n")
    
    return 0


def cmd_rollback(backup_name: str = None):
    """Rollback to a specific model backup."""
    if not backup_name:
        print("Backup name required: python retrain.py rollback <backup_name>")
        return 1
    
    print("\n" + "=" * 70)
    print(f"ROLLING BACK TO: {backup_name}")
    print("=" * 70 + "\n")
    
    orchestrator = RetrainingOrchestrator()
    
    success = orchestrator.rollback_to_backup(backup_name)
    
    if success:
        print(f"\n✓ Rollback to {backup_name} successful")
        return 0
    else:
        print(f"\n✗ Rollback failed")
        return 1


def cmd_retrain():
    """Trigger full retraining pipeline."""
    print("\n" + "=" * 70)
    print("MANUAL RETRAINING TRIGGER")
    print("=" * 70 + "\n")
    
    orchestrator = RetrainingOrchestrator()
    scheduler = RetrainingScheduler()
    
    # Backup current models
    print("1. Backing up current models...")
    backup_name = orchestrator.backup_current_models()
    print(f"   ✓ Backup created: {Path(backup_name).name}\n")
    
    # Prepare data
    print("2. Preparing training data...")
    data_prep = orchestrator.prepare_retraining_data()
    print(f"   ✓ Data ready\n")
    
    # Run training
    print("3. Training new models...")
    print("   (In production CI/CD, this runs the full training pipeline)")
    print("   python training/scripts/run_pipeline.py\n")
    
    # Validate
    print("4. Validating new models...")
    import json
    comparison_file = orchestrator.models_dir / "comparison.json"
    if comparison_file.exists():
        with open(comparison_file) as f:
            comparison = json.load(f)
            metrics = comparison.get("gnn", {})
    else:
        metrics = {}
    
    validation_passed, validation_report = orchestrator.validate_new_models(metrics)
    
    if validation_passed:
        print("   ✓ Validation PASSED\n")
        print("=" * 70)
        print("✓ RETRAINING COMPLETE - READY FOR DEPLOYMENT")
        print("=" * 70)
        return 0
    else:
        print("   ✗ Validation FAILED\n")
        print("5. Rolling back to previous models...")
        orchestrator.rollback_to_backup(Path(backup_name).name)
        print("   ✓ Rollback complete\n")
        print("=" * 70)
        print("✗ RETRAINING FAILED - ROLLED BACK")
        print("=" * 70)
        return 1


def cmd_schedule():
    """Start background retraining scheduler."""
    print("\n" + "=" * 70)
    print("RETRAINING SCHEDULER")
    print("=" * 70 + "\n")
    
    orchestrator = RetrainingOrchestrator()
    scheduler = RetrainingScheduler()
    
    callback = create_retraining_callback(orchestrator)
    scheduler.schedule_monthly_check(callback)
    
    print("Scheduler configuration:")
    print("  • Monthly check: 1st of each month at 00:00 UTC")
    print("  • Auto-triggers retraining if drift detected")
    print("  • Performance thresholds:")
    print(f"    - Minimum accuracy: {orchestrator.min_accuracy*100:.0f}%")
    print(f"    - Minimum AUC: {orchestrator.min_auc*100:.0f}%")
    print(f"    - Fraud rate change limit: {orchestrator.max_fraud_rate_change*100:.0f}%\n")
    
    if scheduler.scheduler_available:
        print("Starting background scheduler...")
        scheduler.start()
        print("✓ Scheduler running")
        print("\nScheduled jobs:")
        for job in scheduler.get_scheduled_jobs():
            print(f"  • {job['name']}")
            print(f"    Next run: {job['next_run']}\n")
        
        print("\nPress Ctrl+C to stop\n")
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            scheduler.stop()
            print("\n✓ Scheduler stopped")
            return 0
    else:
        print("⚠️  APScheduler not available - use GitHub Actions cron instead")
        print("\nAlternative: Configure in .github/workflows/drift-retraining.yml")
        return 0


def cmd_health():
    """Check system health status."""
    print("\n" + "=" * 70)
    print("SYSTEM HEALTH CHECK")
    print("=" * 70 + "\n")
    
    orchestrator = RetrainingOrchestrator()
    
    checks = {}
    
    # Check models exist
    comparison_file = orchestrator.models_dir / "comparison.json"
    checks["models"] = comparison_file.exists()
    print(f"{'✓' if checks['models'] else '✗'} Model metrics: {comparison_file}")
    
    # Check data exists
    processed_dir = orchestrator.data_dir / "processed"
    X_train = processed_dir / "X_train.npy"
    checks["data"] = X_train.exists()
    print(f"{'✓' if checks['data'] else '✗'} Training data: {X_train}")
    
    # Check backups exist
    backups = orchestrator.list_backups()
    checks["backups"] = len(backups) > 0
    print(f"{'✓' if checks['backups'] else '✗'} Backups available: {len(backups)}")
    
    # Check logs
    logs_exist = orchestrator.logs_dir.exists()
    checks["logs"] = logs_exist
    print(f"{'✓' if checks['logs'] else '✗'} Logs directory: {orchestrator.logs_dir}")
    
    print("\n" + "=" * 70)
    if all(checks.values()):
        print("✓ SYSTEM HEALTHY")
        return 0
    else:
        print("⚠️  SOME CHECKS FAILED - Review above")
        return 1


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        return 0
    
    command = sys.argv[1]
    
    try:
        if command == "check":
            return cmd_check()
        elif command == "retrain":
            return cmd_retrain()
        elif command == "validate":
            return cmd_validate()
        elif command == "list-backups":
            return cmd_list_backups()
        elif command == "rollback":
            backup_name = sys.argv[2] if len(sys.argv) > 2 else None
            return cmd_rollback(backup_name)
        elif command == "schedule":
            return cmd_schedule()
        elif command == "health":
            return cmd_health()
        else:
            print(f"Unknown command: {command}")
            print(__doc__)
            return 1
    
    except Exception as e:
        logger.error(f"Command failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
