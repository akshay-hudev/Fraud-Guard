"""
Data Quality Monitoring Module (Step 5)
- Anomaly detection for incoming data
- Distribution drift detection
- Data quality metrics & alerts
- Quality scoring system
"""

from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass
import statistics


@dataclass
class QualityAlert:
    """Data quality alert."""
    severity: str  # "info", "warning", "critical"
    check_type: str  # "anomaly", "drift", "missing", "invalid"
    field: str
    message: str
    value: Any = None
    threshold: Any = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self):
        return {
            "severity": self.severity,
            "check_type": self.check_type,
            "field": self.field,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
            "timestamp": self.timestamp,
        }


class AnomalyDetector:
    """Detect anomalies in numerical features using statistical methods."""
    
    def __init__(self, z_score_threshold: float = 3.0, iqr_multiplier: float = 1.5):
        self.z_score_threshold = z_score_threshold
        self.iqr_multiplier = iqr_multiplier
        self.feature_stats = {}
    
    def fit(self, data: List[Dict[str, Any]], features: List[str]) -> None:
        """Fit anomaly detector on baseline data."""
        for feature in features:
            values = [d.get(feature) for d in data if d.get(feature) is not None]
            if not values:
                continue
            
            # Convert to float if possible
            try:
                values = [float(v) for v in values]
            except (ValueError, TypeError):
                continue
            
            if len(values) < 2:
                continue
            
            mean = statistics.mean(values)
            stdev = statistics.stdev(values) if len(values) > 1 else 0
            sorted_values = sorted(values)
            q1 = sorted_values[len(sorted_values) // 4]
            q3 = sorted_values[3 * len(sorted_values) // 4]
            
            self.feature_stats[feature] = {
                "mean": mean,
                "stdev": stdev,
                "q1": q1,
                "q3": q3,
                "iqr": q3 - q1,
                "min": min(values),
                "max": max(values),
            }
    
    def detect(self, data: Dict[str, Any]) -> List[QualityAlert]:
        """Detect anomalies in a single record."""
        alerts = []
        
        for feature, stats in self.feature_stats.items():
            value = data.get(feature)
            if value is None:
                continue
            
            try:
                value = float(value)
            except (ValueError, TypeError):
                continue
            
            # Z-score method
            if stats["stdev"] > 0:
                z_score = abs((value - stats["mean"]) / stats["stdev"])
                if z_score > self.z_score_threshold:
                    alerts.append(QualityAlert(
                        severity="warning",
                        check_type="anomaly",
                        field=feature,
                        message=f"Z-score anomaly detected: {z_score:.2f}σ",
                        value=value,
                        threshold=self.z_score_threshold,
                    ))
            
            # IQR method
            if stats["iqr"] > 0:
                lower_bound = stats["q1"] - self.iqr_multiplier * stats["iqr"]
                upper_bound = stats["q3"] + self.iqr_multiplier * stats["iqr"]
                
                if value < lower_bound or value > upper_bound:
                    alerts.append(QualityAlert(
                        severity="warning",
                        check_type="anomaly",
                        field=feature,
                        message=f"IQR anomaly: value {value} outside [{lower_bound:.2f}, {upper_bound:.2f}]",
                        value=value,
                        threshold=f"[{lower_bound:.2f}, {upper_bound:.2f}]",
                    ))
        
        return alerts


class DriftDetector:
    """Detect distribution shift in features."""
    
    def __init__(self, drift_threshold: float = 0.2, window_size: int = 100):
        self.drift_threshold = drift_threshold
        self.window_size = window_size
        self.feature_history = defaultdict(list)
        self.baseline_stats = {}
    
    def fit(self, data: List[Dict[str, Any]], features: List[str]) -> None:
        """Fit baseline distribution."""
        for feature in features:
            values = [d.get(feature) for d in data if d.get(feature) is not None]
            if not values:
                continue
            
            try:
                values = [float(v) for v in values]
            except (ValueError, TypeError):
                continue
            
            sorted_vals = sorted(values)
            self.baseline_stats[feature] = {
                "mean": statistics.mean(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0,
                "median": sorted_vals[len(sorted_vals) // 2],
                "q25": sorted_vals[len(sorted_vals) // 4],
                "q75": sorted_vals[3 * len(sorted_vals) // 4],
            }
    
    def detect(self, data: Dict[str, Any]) -> List[QualityAlert]:
        """Detect distribution shifts."""
        alerts = []
        
        for feature, value in data.items():
            if feature not in self.baseline_stats:
                continue
            
            try:
                value = float(value)
            except (ValueError, TypeError):
                continue
            
            # Track history
            self.feature_history[feature].append(value)
            if len(self.feature_history[feature]) > self.window_size:
                self.feature_history[feature] = self.feature_history[feature][-self.window_size:]
            
            # Check drift if we have enough history
            if len(self.feature_history[feature]) >= 20:
                recent_values = self.feature_history[feature][-20:]
                baseline = self.baseline_stats[feature]
                
                recent_mean = statistics.mean(recent_values)
                recent_stdev = statistics.stdev(recent_values) if len(recent_values) > 1 else 0
                
                # Relative change in mean
                if baseline["mean"] != 0:
                    mean_shift = abs(recent_mean - baseline["mean"]) / abs(baseline["mean"])
                    if mean_shift > self.drift_threshold:
                        alerts.append(QualityAlert(
                            severity="warning",
                            check_type="drift",
                            field=feature,
                            message=f"Distribution drift detected: mean shifted {mean_shift*100:.1f}%",
                            value=recent_mean,
                            threshold=baseline["mean"],
                        ))
                
                # Relative change in stdev
                if baseline["stdev"] > 0 and recent_stdev > 0:
                    stdev_shift = abs(recent_stdev - baseline["stdev"]) / baseline["stdev"]
                    if stdev_shift > self.drift_threshold:
                        alerts.append(QualityAlert(
                            severity="info",
                            check_type="drift",
                            field=feature,
                            message=f"Variance shift: stdev changed {stdev_shift*100:.1f}%",
                            value=recent_stdev,
                            threshold=baseline["stdev"],
                        ))
        
        return alerts


class QualityValidator:
    """Validate data quality constraints."""
    
    def __init__(self):
        self.constraints = {}
        self.categorical_values = defaultdict(set)
    
    def set_constraint(self, field: str, min_val: float = None, max_val: float = None,
                      required: bool = False, allowed_values: List[Any] = None) -> None:
        """Set validation constraint for a field."""
        self.constraints[field] = {
            "min": min_val,
            "max": max_val,
            "required": required,
            "allowed_values": set(allowed_values) if allowed_values else None,
        }
    
    def learn_categories(self, data: List[Dict[str, Any]], field: str) -> None:
        """Learn valid categorical values from data."""
        for record in data:
            if field in record and record[field] is not None:
                self.categorical_values[field].add(str(record[field]))
    
    def validate(self, data: Dict[str, Any]) -> List[QualityAlert]:
        """Validate record against constraints."""
        alerts = []
        
        for field, constraint in self.constraints.items():
            value = data.get(field)
            
            # Check required
            if constraint["required"] and (value is None or value == ""):
                alerts.append(QualityAlert(
                    severity="critical",
                    check_type="invalid",
                    field=field,
                    message="Required field is missing",
                ))
                continue
            
            if value is None:
                continue
            
            # Check range
            try:
                numeric_value = float(value)
                if constraint["min"] is not None and numeric_value < constraint["min"]:
                    alerts.append(QualityAlert(
                        severity="warning",
                        check_type="invalid",
                        field=field,
                        message=f"Value {numeric_value} below minimum {constraint['min']}",
                        value=numeric_value,
                        threshold=constraint["min"],
                    ))
                if constraint["max"] is not None and numeric_value > constraint["max"]:
                    alerts.append(QualityAlert(
                        severity="warning",
                        check_type="invalid",
                        field=field,
                        message=f"Value {numeric_value} above maximum {constraint['max']}",
                        value=numeric_value,
                        threshold=constraint["max"],
                    ))
            except (ValueError, TypeError):
                pass
            
            # Check categorical
            if constraint["allowed_values"]:
                if str(value) not in constraint["allowed_values"]:
                    alerts.append(QualityAlert(
                        severity="warning",
                        check_type="invalid",
                        field=field,
                        message=f"Invalid value '{value}' not in allowed set",
                        value=value,
                    ))
        
        return alerts


class QualityMonitor:
    """Main quality monitoring orchestrator."""
    
    def __init__(self):
        self.anomaly_detector = AnomalyDetector()
        self.drift_detector = DriftDetector()
        self.validator = QualityValidator()
        
        self.alerts_history = []
        self.quality_scores = []
        self.feature_stats = {}
    
    def fit(self, data: List[Dict[str, Any]], features: List[str]) -> None:
        """Fit quality monitor on baseline data."""
        self.anomaly_detector.fit(data, features)
        self.drift_detector.fit(data, features)
        
        # Learn categories
        for record in data:
            for feature in features:
                if feature in record:
                    self.validator.learn_categories(data, feature)
        
        # Set default constraints
        for feature in features:
            self.validator.set_constraint(field=feature, required=True)
    
    def check_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check quality of a single record."""
        anomaly_alerts = self.anomaly_detector.detect(data)
        drift_alerts = self.drift_detector.detect(data)
        validation_alerts = self.validator.validate(data)
        
        all_alerts = anomaly_alerts + drift_alerts + validation_alerts
        self.alerts_history.extend(all_alerts)
        
        # Calculate quality score (0-100)
        quality_score = 100
        for alert in all_alerts:
            if alert.severity == "critical":
                quality_score -= 20
            elif alert.severity == "warning":
                quality_score -= 5
            elif alert.severity == "info":
                quality_score -= 1
        
        quality_score = max(0, min(100, quality_score))
        self.quality_scores.append(quality_score)
        
        return {
            "quality_score": round(quality_score, 2),
            "alerts": [a.to_dict() for a in all_alerts],
            "alert_count": len(all_alerts),
            "critical_alerts": sum(1 for a in all_alerts if a.severity == "critical"),
            "warning_alerts": sum(1 for a in all_alerts if a.severity == "warning"),
            "info_alerts": sum(1 for a in all_alerts if a.severity == "info"),
        }
    
    def get_summary(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get quality monitoring summary."""
        if not self.alerts_history:
            return {
                "total_records": 0,
                "avg_quality_score": 100,
                "alerts_count": 0,
                "critical_count": 0,
                "most_common_issues": [],
            }
        
        recent_alerts = [
            a for a in self.alerts_history
            if datetime.fromisoformat(a.timestamp) > datetime.utcnow() - timedelta(minutes=time_window_minutes)
        ]
        
        # Count issues by field
        issue_counts = defaultdict(int)
        for alert in recent_alerts:
            issue_counts[alert.field] += 1
        
        top_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        avg_score = statistics.mean(self.quality_scores[-100:]) if self.quality_scores else 100
        
        return {
            "total_records_checked": len(self.quality_scores),
            "avg_quality_score": round(avg_score, 2),
            "quality_trend": "stable" if avg_score >= 90 else "degrading" if avg_score < 70 else "watch",
            "alerts_count": len(recent_alerts),
            "critical_count": sum(1 for a in recent_alerts if a.severity == "critical"),
            "warning_count": sum(1 for a in recent_alerts if a.severity == "warning"),
            "most_common_issues": [
                {"field": field, "count": count} for field, count in top_issues
            ],
        }
    
    def get_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        return [a.to_dict() for a in self.alerts_history[-limit:]]


# Global instance
quality_monitor = QualityMonitor()
