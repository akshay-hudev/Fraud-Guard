"""
Advanced ML Features Module
- Feature importance ranking & analysis
- Advanced model comparison
- Custom threshold tuning
- Prediction export utilities
"""

import json
import csv
import io
from typing import Dict, List, Any, Tuple
from datetime import datetime
from collections import defaultdict

try:
    import numpy as np
except ImportError:
    np = None
from pathlib import Path


class FeatureImportanceAnalyzer:
    """Aggregate SHAP feature importance from predictions."""
    
    def __init__(self):
        self.feature_impacts = defaultdict(lambda: {"importance": 0, "count": 0})
        self.prediction_history = []
    
    def record_prediction(self, prediction: Dict[str, Any]) -> None:
        """Record a prediction with SHAP values."""
        top_features = prediction.get("top_features", [])
        fraud_score = prediction.get("fraud_score", 0)
        
        for feature_data in top_features:
            feature_name = feature_data.get("feature", "unknown")
            importance = feature_data.get("importance", 0)
            self.feature_impacts[feature_name]["importance"] += importance
            self.feature_impacts[feature_name]["count"] += 1
        
        self.prediction_history.append({
            "timestamp": datetime.now().isoformat(),
            "fraud_score": fraud_score,
            "top_features": top_features,
        })
    
    def get_feature_importance_rank(self, top_n: int = 15) -> List[Dict[str, Any]]:
        """Get ranked feature importance across all predictions."""
        ranked = []
        for feature, data in self.feature_impacts.items():
            avg_importance = data["importance"] / max(data["count"], 1)
            ranked.append({
                "feature": feature,
                "avg_importance": round(avg_importance, 4),
                "occurrences": data["count"],
                "total_impact": round(data["importance"], 4),
            })
        
        ranked.sort(key=lambda x: x["avg_importance"], reverse=True)
        return ranked[:top_n]
    
    def feature_stats_by_fraud_score(self, bin_size: float = 0.1) -> Dict[str, List[Dict]]:
        """Analyze feature importance across different fraud score ranges."""
        bins = defaultdict(list)
        for pred in self.prediction_history:
            fraud_score = pred["fraud_score"]
            bin_label = f"{int(fraud_score / bin_size) * bin_size:.1f}-{(int(fraud_score / bin_size) + 1) * bin_size:.1f}"
            bins[bin_label].extend(pred["top_features"])
        
        stats = {}
        for bin_label, features in bins.items():
            feature_counts = defaultdict(int)
            for feature_data in features:
                feature_counts[feature_data.get("feature", "unknown")] += 1
            stats[bin_label] = [
                {"feature": f, "count": c} for f, c in sorted(
                    feature_counts.items(), key=lambda x: x[1], reverse=True
                )[:5]
            ]
        return stats


class ModelComparator:
    """Advanced model comparison and analysis."""
    
    @staticmethod
    def compare_models(models_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Compare models across multiple metrics."""
        comparison = {
            "models": {},
            "rankings": {},
            "summary": {
                "best_overall": None,
                "best_by_metric": {},
            },
        }
        
        # Build model info
        for model_name, metrics in models_metrics.items():
            comparison["models"][model_name] = {
                "metrics": metrics,
                "weighted_score": sum(metrics.values()) / len(metrics) if metrics else 0,
            }
        
        # Find best overall (highest weighted score)
        best_model = max(
            comparison["models"].items(),
            key=lambda x: x[1]["weighted_score"],
        )[0]
        comparison["summary"]["best_overall"] = best_model
        
        # Find best per metric
        all_metrics = set()
        for metrics in models_metrics.values():
            all_metrics.update(metrics.keys())
        
        for metric in all_metrics:
            best = max(
                models_metrics.items(),
                key=lambda x: x[1].get(metric, 0),
            )
            comparison["summary"]["best_by_metric"][metric] = {
                "model": best[0],
                "value": round(best[1].get(metric, 0), 4),
            }
        
        # Generate rankings
        for metric in all_metrics:
            ranked = sorted(
                [(m, metrics.get(metric, 0)) for m, metrics in models_metrics.items()],
                key=lambda x: x[1],
                reverse=True,
            )
            comparison["rankings"][metric] = [
                {"rank": i + 1, "model": m, "value": round(v, 4)}
                for i, (m, v) in enumerate(ranked)
            ]
        
        return comparison
    
    @staticmethod
    def calculate_trading_metrics(
        predictions_true: List[float],
        predictions_pred: List[float],
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Calculate precision, recall, F1, AUC."""
        pred_binary = [1 if p >= threshold else 0 for p in predictions_pred]
        true_binary = [1 if p >= 0.5 else 0 for p in predictions_true]
        
        tp = sum(1 for t, p in zip(true_binary, pred_binary) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(true_binary, pred_binary) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(true_binary, pred_binary) if t == 1 and p == 0)
        tn = sum(1 for t, p in zip(true_binary, pred_binary) if t == 0 and p == 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(true_binary) if len(true_binary) > 0 else 0
        
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }


class ThresholdOptimizer:
    """Optimize fraud detection thresholds for different use cases."""
    
    @staticmethod
    def find_optimal_thresholds(
        predictions: List[float],
        ground_truth: List[float] = None,
        use_case: str = "balanced",
    ) -> Dict[str, Any]:
        """Find optimal thresholds based on use case."""
        thresholds = [i / 100 for i in range(0, 101, 5)]
        results = []
        
        for threshold in thresholds:
            metrics = {
                "threshold": round(threshold, 2),
                "positive_rate": round(sum(1 for p in predictions if p >= threshold) / len(predictions), 2),
            }
            
            if ground_truth:
                comp_metrics = ModelComparator.calculate_trading_metrics(
                    ground_truth, predictions, threshold
                )
                metrics.update(comp_metrics)
            
            results.append(metrics)
        
        # Filter by use case
        if use_case == "conservative":  # Minimize false positives
            best = max(results, key=lambda x: x.get("precision", 0))
        elif use_case == "aggressive":  # Minimize false negatives
            best = max(results, key=lambda x: x.get("recall", 0))
        else:  # Balanced (F1)
            best = max(results, key=lambda x: x.get("f1", 0))
        
        return {
            "recommended_threshold": best["threshold"],
            "use_case": use_case,
            "use_case_metrics": best,
            "all_thresholds": results,
        }


class PredictionExporter:
    """Export predictions in various formats."""
    
    @staticmethod
    def export_to_csv(predictions: List[Dict[str, Any]]) -> str:
        """Export predictions to CSV string."""
        if not predictions:
            return ""
        
        output = io.StringIO()
        fieldnames = [
            "prediction_id", "claim_id", "fraud_score", "fraud_prediction",
            "confidence", "inference_time_ms", "model_version",
        ]
        
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        for pred in predictions:
            writer.writerow({
                "prediction_id": pred.get("prediction_id", ""),
                "claim_id": pred.get("claim_id", ""),
                "fraud_score": pred.get("fraud_score", ""),
                "fraud_prediction": pred.get("fraud_prediction", ""),
                "confidence": pred.get("confidence", ""),
                "inference_time_ms": pred.get("inference_time_ms", ""),
                "model_version": pred.get("model_version", ""),
            })
        
        return output.getvalue()
    
    @staticmethod
    def export_to_json(predictions: List[Dict[str, Any]]) -> str:
        """Export predictions to JSON string."""
        return json.dumps({
            "exported_at": datetime.now().isoformat(),
            "total_predictions": len(predictions),
            "predictions": predictions,
        }, indent=2)
    
    @staticmethod
    def export_summary(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not predictions:
            return {
                "total": 0,
                "fraud_count": 0,
                "legitimate_count": 0,
                "avg_fraud_score": 0,
                "fraud_rate": 0,
            }
        
        fraud_count = sum(1 for p in predictions if p.get("fraud_prediction", False))
        scores = [p.get("fraud_score", 0) for p in predictions]
        latencies = [p.get("inference_time_ms", 0) for p in predictions]
        return {
            "total": len(predictions),
            "fraud_count": fraud_count,
            "legitimate_count": len(predictions) - fraud_count,
            "avg_fraud_score": round(sum(scores) / len(scores), 4) if scores else 0,
            "fraud_rate": round(fraud_count / len(predictions), 4),
            "avg_inference_time_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        }


class BatchJobTracker:
    """Track batch upload/processing jobs."""
    
    def __init__(self):
        self.jobs = {}
    
    def create_job(self, job_id: str, total_items: int) -> Dict[str, Any]:
        """Create new batch job."""
        self.jobs[job_id] = {
            "job_id": job_id,
            "status": "processing",
            "total_items": total_items,
            "processed_items": 0,
            "successful_items": 0,
            "failed_items": 0,
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
            "errors": [],
            "predictions": [],
        }
        return self.jobs[job_id]
    
    def update_job_progress(
        self,
        job_id: str,
        processed: int,
        successful: int,
        failed: int,
        prediction: Dict = None,
        error: str = None,
    ) -> Dict[str, Any]:
        """Update job progress."""
        if job_id not in self.jobs:
            return None
        
        job = self.jobs[job_id]
        job["processed_items"] = processed
        job["successful_items"] = successful
        job["failed_items"] = failed
        
        if prediction:
            job["predictions"].append(prediction)
        
        if error:
            job["errors"].append({
                "timestamp": datetime.now().isoformat(),
                "error": error,
            })
        
        # Auto-complete if all processed
        if processed >= job["total_items"]:
            job["status"] = "completed"
            job["completed_at"] = datetime.now().isoformat()
        
        return job
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status."""
        if job_id not in self.jobs:
            return {"error": f"Job {job_id} not found"}
        
        job = self.jobs[job_id]
        progress_pct = (job["processed_items"] / max(job["total_items"], 1)) * 100
        
        return {
            "job_id": job_id,
            "status": job["status"],
            "progress": round(progress_pct, 1),
            "processed": job["processed_items"],
            "total": job["total_items"],
            "successful": job["successful_items"],
            "failed": job["failed_items"],
            "error_count": len(job["errors"]),
            "started_at": job["started_at"],
            "completed_at": job["completed_at"],
            "predictions_count": len(job["predictions"]),
        }
    
    def get_job_results(self, job_id: str) -> Dict[str, Any]:
        """Get full job results."""
        if job_id not in self.jobs:
            return {"error": f"Job {job_id} not found"}
        
        job = self.jobs[job_id]
        summary = PredictionExporter.export_summary(job["predictions"])
        
        return {
            "status": self.get_job_status(job_id),
            "summary": summary,
            "predictions": job["predictions"][:100],  # Return first 100
            "total_predictions": len(job["predictions"]),
            "errors": job["errors"][:10],  # Return first 10 errors
        }


# Global instances
feature_analyzer = FeatureImportanceAnalyzer()
batch_tracker = BatchJobTracker()
