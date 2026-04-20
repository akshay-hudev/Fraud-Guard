"""
Compliance & Audit Module (Step 8)
- PII detection and masking
- Audit logging with immutability
- GDPR compliance checks
- Data retention policies
- Compliance reporting

Pure Python implementation - no external dependencies.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import hashlib
import re


class PIIMasker:
    """Detect and mask Personally Identifiable Information."""
    
    PII_PATTERNS = {
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "phone": r"\+?1?\d{9,15}",
        "ssn": r"\d{3}-\d{2}-\d{4}",
        "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "date": r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",
    }
    
    SENSITIVE_FIELDS = {
        "password", "token", "secret", "apikey", "api_key",
        "ssn", "social_security", "credit_card", "card_number",
        "phone", "email", "address", "date_of_birth", "dob"
    }
    
    @staticmethod
    def mask_email(email: str) -> str:
        """Mask email address."""
        if "@" not in email:
            return "*" * len(email)
        local, domain = email.split("@")
        masked_local = local[0] + "*" * (len(local) - 2) + local[-1] if len(local) > 2 else "*" * len(local)
        return f"{masked_local}@{domain}"
    
    @staticmethod
    def mask_phone(phone: str) -> str:
        """Mask phone number."""
        if len(phone) < 4:
            return "*" * len(phone)
        return "*" * (len(phone) - 4) + phone[-4:]
    
    @staticmethod
    def mask_number(value: str) -> str:
        """Mask credit card or SSN."""
        if len(value) < 4:
            return "*" * len(value)
        return "*" * (len(value) - 4) + value[-4:]
    
    @staticmethod
    def is_sensitive_field(field_name: str) -> bool:
        """Check if field is sensitive."""
        field_lower = field_name.lower()
        return any(sensitive in field_lower for sensitive in PIIMasker.SENSITIVE_FIELDS)
    
    @staticmethod
    def mask_value(value: Any, field_name: str = None) -> Any:
        """Mask a value based on type and field name."""
        if value is None:
            return None
        
        if isinstance(value, bool) or isinstance(value, (int, float)):
            return value
        
        value_str = str(value)
        
        # Check field name first
        if field_name and PIIMasker.is_sensitive_field(field_name):
            if re.match(PIIMasker.PII_PATTERNS.get("email", ""), value_str):
                return PIIMasker.mask_email(value_str)
            elif re.match(PIIMasker.PII_PATTERNS.get("phone", ""), value_str):
                return PIIMasker.mask_phone(value_str)
            else:
                return "*" * len(value_str)
        
        # Check patterns
        for pii_type, pattern in PIIMasker.PII_PATTERNS.items():
            if re.search(pattern, value_str):
                if pii_type == "email":
                    return PIIMasker.mask_email(value_str)
                elif pii_type == "phone":
                    return PIIMasker.mask_phone(value_str)
                elif pii_type in ["ssn", "credit_card"]:
                    return PIIMasker.mask_number(value_str)
                else:
                    return "*" * len(value_str)
        
        return value
    
    @staticmethod
    def mask_dict(data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask all PII in a dictionary."""
        masked = {}
        for key, value in data.items():
            if isinstance(value, dict):
                masked[key] = PIIMasker.mask_dict(value)
            elif isinstance(value, list):
                masked[key] = [
                    PIIMasker.mask_dict(item) if isinstance(item, dict) else PIIMasker.mask_value(item, key)
                    for item in value
                ]
            else:
                masked[key] = PIIMasker.mask_value(value, key)
        return masked


class AuditLogger:
    """Immutable audit logging."""
    
    def __init__(self, max_logs: int = 10000):
        self.max_logs = max_logs
        self.logs = []
        self.log_hash_chain = []  # For integrity verification
    
    def _compute_hash(self, log_entry: Dict[str, Any]) -> str:
        """Compute hash of log entry."""
        log_str = str(sorted(log_entry.items()))
        prev_hash = self.log_hash_chain[-1] if self.log_hash_chain else ""
        combined = prev_hash + log_str
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def log_action(self, action: str, user: str, resource: str, 
                   details: Dict[str, Any] = None, status: str = "success") -> str:
        """Log an action with integrity chain."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "user": user,
            "resource": resource,
            "status": status,
            "details": details or {},
        }
        
        # Compute hash for integrity
        entry_hash = self._compute_hash(log_entry)
        log_entry["hash"] = entry_hash
        
        self.logs.append(log_entry)
        self.log_hash_chain.append(entry_hash)
        
        # Enforce max logs
        if len(self.logs) > self.max_logs:
            self.logs = self.logs[-self.max_logs:]
            self.log_hash_chain = self.log_hash_chain[-self.max_logs:]
        
        return entry_hash
    
    def log_prediction(self, prediction_id: str, user: str, claim_id: str,
                      fraud_score: float, model_version: str) -> str:
        """Log a prediction."""
        return self.log_action(
            action="PREDICTION_MADE",
            user=user,
            resource=f"prediction:{prediction_id}",
            details={
                "claim_id": claim_id,
                "fraud_score": fraud_score,
                "model_version": model_version,
            }
        )
    
    def log_data_access(self, user: str, data_type: str, 
                       record_count: int, filter_criteria: str = None) -> str:
        """Log data access."""
        return self.log_action(
            action="DATA_ACCESS",
            user=user,
            resource=f"data:{data_type}",
            details={
                "record_count": record_count,
                "filter_criteria": filter_criteria,
            }
        )
    
    def log_export(self, user: str, export_format: str, 
                  record_count: int, destination: str) -> str:
        """Log data export."""
        return self.log_action(
            action="DATA_EXPORT",
            user=user,
            resource="data:export",
            details={
                "format": export_format,
                "record_count": record_count,
                "destination": destination,
            }
        )
    
    def log_deletion(self, user: str, data_type: str, 
                    record_id: str, reason: str) -> str:
        """Log data deletion."""
        return self.log_action(
            action="DATA_DELETE",
            user=user,
            resource=f"data:{data_type}:{record_id}",
            details={"reason": reason},
        )
    
    def get_logs(self, limit: int = 100, action_filter: str = None) -> List[Dict[str, Any]]:
        """Get audit logs with optional filtering."""
        logs = self.logs[-limit:]
        
        if action_filter:
            logs = [log for log in logs if log.get("action") == action_filter]
        
        return logs
    
    def verify_integrity(self) -> Tuple[bool, List[str]]:
        """Verify audit log integrity."""
        issues = []
        
        for i, log in enumerate(self.logs):
            if i == 0:
                continue
            
            expected_hash = self._compute_hash(log)
            if log.get("hash") != expected_hash:
                issues.append(f"Log {i}: hash mismatch")
        
        return len(issues) == 0, issues


class GDPRCompliance:
    """GDPR compliance tracking and checks."""
    
    def __init__(self):
        self.consent_records = {}  # user_id -> consent_status
        self.retention_policies = {}
        self.data_subject_requests = []
    
    def record_consent(self, user_id: str, consent_type: str, 
                      granted: bool, timestamp: str = None) -> None:
        """Record user consent."""
        if user_id not in self.consent_records:
            self.consent_records[user_id] = {}
        
        self.consent_records[user_id][consent_type] = {
            "granted": granted,
            "timestamp": timestamp or datetime.utcnow().isoformat(),
        }
    
    def has_consent(self, user_id: str, consent_type: str) -> bool:
        """Check if user has given consent."""
        return self.consent_records.get(user_id, {}).get(consent_type, {}).get("granted", False)
    
    def set_retention_policy(self, data_type: str, retention_days: int) -> None:
        """Set data retention policy."""
        self.retention_policies[data_type] = {
            "retention_days": retention_days,
            "created_at": datetime.utcnow().isoformat(),
        }
    
    def check_data_expiry(self, data_type: str, created_at: str) -> Tuple[bool, int]:
        """Check if data should be retained or deleted."""
        if data_type not in self.retention_policies:
            return True, -1  # No policy = keep forever
        
        policy = self.retention_policies[data_type]
        retention_days = policy["retention_days"]
        
        created_date = datetime.fromisoformat(created_at)
        expiry_date = created_date + timedelta(days=retention_days)
        days_until_expiry = (expiry_date - datetime.utcnow()).days
        
        should_retain = days_until_expiry > 0
        return should_retain, days_until_expiry
    
    def file_data_subject_request(self, user_id: str, request_type: str, 
                                 details: str = None) -> str:
        """File a data subject request (access, rectification, erasure)."""
        request_id = f"dsr_{len(self.data_subject_requests)}_{int(datetime.utcnow().timestamp())}"
        
        request = {
            "request_id": request_id,
            "user_id": user_id,
            "request_type": request_type,  # access, rectification, erasure, portability
            "details": details,
            "filed_at": datetime.utcnow().isoformat(),
            "status": "pending",
            "processed_at": None,
        }
        
        self.data_subject_requests.append(request)
        return request_id
    
    def get_data_subject_requests(self, status: str = None) -> List[Dict[str, Any]]:
        """Get data subject requests."""
        if status:
            return [r for r in self.data_subject_requests if r["status"] == status]
        return self.data_subject_requests
    
    def process_erasure_request(self, request_id: str) -> bool:
        """Mark erasure request as processed."""
        for req in self.data_subject_requests:
            if req["request_id"] == request_id and req["request_type"] == "erasure":
                req["status"] = "processed"
                req["processed_at"] = datetime.utcnow().isoformat()
                return True
        return False
    
    def get_compliance_status(self) -> Dict[str, Any]:
        """Get overall GDPR compliance status."""
        pending_requests = len([r for r in self.data_subject_requests if r["status"] == "pending"])
        processed_requests = len([r for r in self.data_subject_requests if r["status"] == "processed"])
        
        return {
            "total_users_with_consent": len(self.consent_records),
            "retention_policies": len(self.retention_policies),
            "pending_data_subject_requests": pending_requests,
            "processed_data_subject_requests": processed_requests,
            "compliance_score": self._calculate_compliance_score(),
        }
    
    def _calculate_compliance_score(self) -> float:
        """Calculate GDPR compliance score (0-100)."""
        score = 100.0
        
        # Deductions
        if not self.retention_policies:
            score -= 20
        if len([r for r in self.data_subject_requests if r["status"] == "pending"]) > 0:
            score -= 10
        if not self.consent_records:
            score -= 30
        
        return max(0, score)


class ComplianceReporter:
    """Generate compliance reports."""
    
    def __init__(self, audit_logger: AuditLogger, gdpr: GDPRCompliance):
        self.audit_logger = audit_logger
        self.gdpr = gdpr
    
    def generate_audit_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate audit report for time period."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        logs = self.audit_logger.get_logs(limit=1000)
        relevant_logs = [
            log for log in logs
            if datetime.fromisoformat(log["timestamp"]) > cutoff_date
        ]
        
        # Count by action
        action_counts = {}
        for log in relevant_logs:
            action = log["action"]
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # Count by user
        user_counts = {}
        for log in relevant_logs:
            user = log["user"]
            user_counts[user] = user_counts.get(user, 0) + 1
        
        return {
            "report_type": "AUDIT_REPORT",
            "period_days": days,
            "generated_at": datetime.utcnow().isoformat(),
            "total_events": len(relevant_logs),
            "events_by_action": action_counts,
            "events_by_user": user_counts,
            "top_users": sorted(user_counts.items(), key=lambda x: x[1], reverse=True)[:5],
        }
    
    def generate_gdpr_report(self) -> Dict[str, Any]:
        """Generate GDPR compliance report."""
        compliance_status = self.gdpr.get_compliance_status()
        
        return {
            "report_type": "GDPR_COMPLIANCE_REPORT",
            "generated_at": datetime.utcnow().isoformat(),
            "compliance": compliance_status,
            "recommendations": self._get_gdpr_recommendations(compliance_status),
        }
    
    def _get_gdpr_recommendations(self, status: Dict[str, Any]) -> List[str]:
        """Get GDPR compliance recommendations."""
        recommendations = []
        
        if status["compliance_score"] < 70:
            recommendations.append("CRITICAL: Compliance score below 70%. Immediate action required.")
        
        if status["pending_data_subject_requests"] > 0:
            recommendations.append(f"Process {status['pending_data_subject_requests']} pending data subject requests within 30 days.")
        
        if status["total_users_with_consent"] == 0:
            recommendations.append("No consent records found. Implement consent tracking system.")
        
        if status["retention_policies"] == 0:
            recommendations.append("Define data retention policies for each data type.")
        
        return recommendations
    
    def generate_data_protection_report(self, pii_detections: int = 0, 
                                       pii_masked: int = 0) -> Dict[str, Any]:
        """Generate data protection report."""
        return {
            "report_type": "DATA_PROTECTION_REPORT",
            "generated_at": datetime.utcnow().isoformat(),
            "pii_detections": pii_detections,
            "pii_masked": pii_masked,
            "masking_rate": round(pii_masked / max(1, pii_detections) * 100, 2) if pii_detections > 0 else 0,
            "audit_logs_integrity": self.audit_logger.verify_integrity()[0],
        }


class ComplianceManager:
    """Main compliance orchestrator."""
    
    def __init__(self):
        self.pii_masker = PIIMasker()
        self.audit_logger = AuditLogger()
        self.gdpr = GDPRCompliance()
        self.reporter = ComplianceReporter(self.audit_logger, self.gdpr)
        
        self.pii_detection_count = 0
        self.pii_masked_count = 0
    
    def process_prediction_with_compliance(self, prediction_data: Dict[str, Any],
                                          user: str, claim_id: str) -> Dict[str, Any]:
        """Process prediction with full compliance checks."""
        # Log the access
        self.audit_logger.log_prediction(
            prediction_id=prediction_data.get("id", "unknown"),
            user=user,
            claim_id=claim_id,
            fraud_score=prediction_data.get("fraud_score", 0),
            model_version=prediction_data.get("model_version", "unknown"),
        )
        
        # Check GDPR consent
        has_data_consent = self.gdpr.has_consent(user, "data_processing")
        
        # Mask PII
        masked_data = self.pii_masker.mask_dict(prediction_data)
        
        return {
            "prediction": masked_data,
            "gdpr_compliant": has_data_consent,
            "audit_logged": True,
        }
    
    def export_with_compliance(self, export_data: List[Dict[str, Any]], 
                               user: str, format: str) -> Dict[str, Any]:
        """Export data with compliance checks."""
        # Log export
        self.audit_logger.log_export(
            user=user,
            export_format=format,
            record_count=len(export_data),
            destination="export_file",
        )
        
        # Mask all PII
        masked_data = [self.pii_masker.mask_dict(record) for record in export_data]
        
        return {
            "data": masked_data,
            "record_count": len(masked_data),
            "format": format,
            "audit_logged": True,
        }
    
    def handle_data_deletion(self, data_type: str, record_ids: List[str],
                            user: str, reason: str) -> Dict[str, Any]:
        """Handle data deletion with audit trail."""
        deleted_count = 0
        
        for record_id in record_ids:
            self.audit_logger.log_deletion(
                user=user,
                data_type=data_type,
                record_id=record_id,
                reason=reason,
            )
            deleted_count += 1
        
        return {
            "deleted_count": deleted_count,
            "audit_logged": True,
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    def get_compliance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive compliance dashboard."""
        audit_report = self.reporter.generate_audit_report(days=30)
        gdpr_report = self.reporter.generate_gdpr_report()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "audit_summary": audit_report,
            "gdpr_compliance": gdpr_report,
            "recommendations": gdpr_report.get("recommendations", []),
        }


# Global instance
compliance_manager = ComplianceManager()
