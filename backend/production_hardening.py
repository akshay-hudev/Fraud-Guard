"""
Production Hardening Module: Circuit Breaker, Failover, Rate Limiting, Graceful Degradation
Step 10: Enterprise-grade resilience and fault tolerance
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Any, Optional
from enum import Enum
from threading import Lock
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"          # Normal operation
    OPEN = "open"              # Failing, reject requests
    HALF_OPEN = "half_open"    # Testing if recovered


class HealthStatus(Enum):
    """Service health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5           # Failures before opening
    success_threshold: int = 2           # Successes before closing (from half-open)
    timeout_secs: int = 60               # Time before half-open attempt
    name: str = "CircuitBreaker"


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_changes: int = 0
    last_failure_time: Optional[datetime] = None
    last_state_change: datetime = field(default_factory=datetime.utcnow)


class CircuitBreaker:
    """Circuit Breaker pattern: prevent cascading failures."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats()
        self.failure_count = 0
        self.success_count = 0
        self.lock = Lock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self.lock:
            if self.state == CircuitState.OPEN:
                # Check if timeout expired (try half-open)
                time_since_open = datetime.utcnow() - self.stats.last_state_change
                if time_since_open.total_seconds() >= self.config.timeout_secs:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info(f"Circuit breaker {self.config.name} → HALF_OPEN")
                    self.stats.state_changes += 1
                else:
                    # Still open, reject
                    self.stats.rejected_calls += 1
                    raise Exception(
                        f"Circuit breaker {self.config.name} is OPEN. "
                        f"Service unavailable. Retry in {self.config.timeout_secs}s"
                    )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful call."""
        with self.lock:
            self.stats.successful_calls += 1
            self.stats.total_calls += 1
            self.failure_count = 0
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                
                # Close if threshold reached
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    logger.info(f"Circuit breaker {self.config.name} → CLOSED (recovered)")
                    self.stats.state_changes += 1
                    self.success_count = 0
    
    def _on_failure(self):
        """Handle failed call."""
        with self.lock:
            self.stats.failed_calls += 1
            self.stats.total_calls += 1
            self.stats.last_failure_time = datetime.utcnow()
            self.failure_count += 1
            self.success_count = 0
            
            # Open if threshold reached
            if self.failure_count >= self.config.failure_threshold:
                if self.state != CircuitState.OPEN:
                    self.state = CircuitState.OPEN
                    self.stats.last_state_change = datetime.utcnow()
                    logger.warning(
                        f"Circuit breaker {self.config.name} → OPEN "
                        f"({self.failure_count} failures)"
                    )
                    self.stats.state_changes += 1
                    self.failure_count = 0
    
    def get_status(self) -> Dict:
        """Get circuit breaker status."""
        with self.lock:
            return {
                "name": self.config.name,
                "state": self.state.value,
                "total_calls": self.stats.total_calls,
                "successful": self.stats.successful_calls,
                "failed": self.stats.failed_calls,
                "rejected": self.stats.rejected_calls,
                "success_rate": round(
                    self.stats.successful_calls / max(self.stats.total_calls, 1), 3
                ),
                "state_changes": self.stats.state_changes,
                "last_failure": self.stats.last_failure_time.isoformat() 
                               if self.stats.last_failure_time else None
            }


class FailoverManager:
    """Failover & Redundancy: switch to backup services."""
    
    def __init__(self):
        self.primary_service: Optional[Callable] = None
        self.backup_services: List[Callable] = []
        self.current_index = 0
        self.failover_count = 0
        self.lock = Lock()
    
    def register_services(self, primary: Callable, backups: List[Callable] = None):
        """Register primary and backup services."""
        self.primary_service = primary
        self.backup_services = backups or []
    
    def call_with_failover(self, *args, **kwargs) -> Any:
        """Call with automatic failover to backups."""
        # Try primary first
        if self.primary_service:
            try:
                return self.primary_service(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Primary service failed: {e}. Trying backups...")
        
        # Try backups
        for i, backup in enumerate(self.backup_services):
            try:
                logger.info(f"Attempting backup service {i+1}...")
                result = backup(*args, **kwargs)
                
                with self.lock:
                    self.failover_count += 1
                
                return result
            except Exception as e:
                logger.warning(f"Backup {i+1} failed: {e}")
        
        # All failed
        raise Exception("All services failed. No failover available.")
    
    def get_failover_stats(self) -> Dict:
        """Get failover statistics."""
        with self.lock:
            return {
                "failovers_triggered": self.failover_count,
                "primary_available": self.primary_service is not None,
                "backup_count": len(self.backup_services)
            }


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, rate: int, per_seconds: int = 1):
        """
        Create rate limiter.
        
        Args:
            rate: Tokens per period (e.g., 100 requests)
            per_seconds: Time period in seconds (e.g., 1 for per-second)
        """
        self.rate = rate
        self.per_seconds = per_seconds
        self.tokens = rate
        self.last_refill = datetime.utcnow()
        self.lock = Lock()
        self.stats = {
            "allowed": 0,
            "rejected": 0,
            "refills": 0
        }
    
    def allow_request(self, tokens_needed: int = 1) -> bool:
        """Check if request is allowed (token available)."""
        with self.lock:
            self._refill_tokens()
            
            if self.tokens >= tokens_needed:
                self.tokens -= tokens_needed
                self.stats["allowed"] += 1
                return True
            else:
                self.stats["rejected"] += 1
                return False
    
    def _refill_tokens(self):
        """Refill tokens based on elapsed time."""
        now = datetime.utcnow()
        elapsed = (now - self.last_refill).total_seconds()
        
        if elapsed >= self.per_seconds:
            refill_amount = int(elapsed / self.per_seconds) * self.rate
            self.tokens = min(self.rate, self.tokens + refill_amount)
            self.last_refill = now
            self.stats["refills"] += 1
    
    def get_stats(self) -> Dict:
        """Get rate limiter stats."""
        with self.lock:
            total = self.stats["allowed"] + self.stats["rejected"]
            return {
                "tokens_available": self.tokens,
                "requests_allowed": self.stats["allowed"],
                "requests_rejected": self.stats["rejected"],
                "rejection_rate": round(
                    self.stats["rejected"] / max(total, 1), 3
                ),
                "refills": self.stats["refills"]
            }


class BulkheadPattern:
    """Bulkhead pattern: isolate resources and limit concurrency."""
    
    def __init__(self, name: str, max_concurrent: int = 10):
        self.name = name
        self.max_concurrent = max_concurrent
        self.current_tasks = 0
        self.total_tasks = 0
        self.rejected_tasks = 0
        self.lock = Lock()
    
    def acquire(self) -> bool:
        """Try to acquire a resource slot."""
        with self.lock:
            if self.current_tasks < self.max_concurrent:
                self.current_tasks += 1
                self.total_tasks += 1
                return True
            else:
                self.rejected_tasks += 1
                return False
    
    def release(self):
        """Release a resource slot."""
        with self.lock:
            self.current_tasks = max(0, self.current_tasks - 1)
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with bulkhead protection."""
        if not self.acquire():
            raise Exception(
                f"Bulkhead {self.name} at capacity. "
                f"Current: {self.current_tasks}/{self.max_concurrent}"
            )
        
        try:
            return func(*args, **kwargs)
        finally:
            self.release()
    
    def get_status(self) -> Dict:
        """Get bulkhead status."""
        with self.lock:
            return {
                "name": self.name,
                "current_tasks": self.current_tasks,
                "max_concurrent": self.max_concurrent,
                "utilization": round(self.current_tasks / self.max_concurrent, 3),
                "total_executed": self.total_tasks,
                "rejected": self.rejected_tasks
            }


class HealthChecker:
    """Health checks with self-healing capabilities."""
    
    def __init__(self, check_interval_secs: int = 30):
        self.check_interval = check_interval_secs
        self.checks = {}  # name -> (check_func, last_result, last_check_time)
        self.health_history = {}  # name -> [status, status, ...]
        self.lock = Lock()
    
    def register_check(self, name: str, check_func: Callable):
        """Register a health check."""
        with self.lock:
            self.checks[name] = {
                "func": check_func,
                "last_result": None,
                "last_check_time": None,
                "consecutive_failures": 0
            }
            self.health_history[name] = []
    
    def run_checks(self) -> Dict[str, str]:
        """Run all health checks."""
        results = {}
        
        with self.lock:
            for name, check_info in self.checks.items():
                now = datetime.utcnow()
                
                # Skip if checked recently
                last_check = check_info["last_check_time"]
                if last_check and (now - last_check).total_seconds() < self.check_interval:
                    results[name] = check_info["last_result"]
                    continue
                
                try:
                    is_healthy = check_info["func"]()
                    status = HealthStatus.HEALTHY.value if is_healthy else HealthStatus.DEGRADED.value
                    check_info["last_result"] = status
                    check_info["last_check_time"] = now
                    check_info["consecutive_failures"] = 0
                except Exception as e:
                    check_info["consecutive_failures"] += 1
                    
                    if check_info["consecutive_failures"] >= 3:
                        status = HealthStatus.UNHEALTHY.value
                    else:
                        status = HealthStatus.DEGRADED.value
                    
                    check_info["last_result"] = status
                    check_info["last_check_time"] = now
                    logger.error(f"Health check {name} failed: {e}")
                
                results[name] = status
                self.health_history[name].append({
                    "status": status,
                    "timestamp": now.isoformat()
                })
        
        return results
    
    def get_overall_health(self) -> str:
        """Get overall system health."""
        results = self.run_checks()
        
        if all(v == HealthStatus.HEALTHY.value for v in results.values()):
            return HealthStatus.HEALTHY.value
        elif any(v == HealthStatus.UNHEALTHY.value for v in results.values()):
            return HealthStatus.UNHEALTHY.value
        else:
            return HealthStatus.DEGRADED.value
    
    def get_health_status(self) -> Dict:
        """Get detailed health status."""
        results = self.run_checks()
        
        return {
            "overall": self.get_overall_health(),
            "checks": results,
            "timestamp": datetime.utcnow().isoformat()
        }


class GracefulDegradation:
    """Graceful degradation: degrade features instead of failing."""
    
    def __init__(self):
        self.features = {}  # name -> enabled
        self.feature_dependencies = {}  # feature -> list of dependent features
        self.disabled_features = set()
        self.lock = Lock()
    
    def register_feature(self, name: str, dependencies: List[str] = None):
        """Register a feature."""
        with self.lock:
            self.features[name] = True
            self.feature_dependencies[name] = dependencies or []
    
    def disable_feature(self, name: str, reason: str = "") -> bool:
        """Disable a feature and cascade to dependents."""
        with self.lock:
            if name not in self.features:
                return False
            
            self.features[name] = False
            self.disabled_features.add(name)
            
            # Disable dependent features
            for feat_name, deps in self.feature_dependencies.items():
                if name in deps and self.features.get(feat_name, True):
                    self.features[feat_name] = False
                    self.disabled_features.add(feat_name)
                    logger.warning(f"Cascaded disable: {feat_name} (depends on {name}). Reason: {reason}")
            
            logger.warning(f"Feature disabled: {name}. Reason: {reason}")
            return True
    
    def enable_feature(self, name: str) -> bool:
        """Enable a feature if dependencies are met."""
        with self.lock:
            if name not in self.features:
                return False
            
            deps = self.feature_dependencies.get(name, [])
            
            # Check if dependencies are enabled
            for dep in deps:
                if dep in self.disabled_features:
                    logger.warning(f"Cannot enable {name}: dependency {dep} is disabled")
                    return False
            
            self.features[name] = True
            self.disabled_features.discard(name)
            logger.info(f"Feature enabled: {name}")
            return True
    
    def is_enabled(self, name: str) -> bool:
        """Check if feature is enabled."""
        return self.features.get(name, False)
    
    def get_status(self) -> Dict:
        """Get degradation status."""
        with self.lock:
            return {
                "total_features": len(self.features),
                "enabled": sum(1 for v in self.features.values() if v),
                "disabled": len(self.disabled_features),
                "disabled_features": list(self.disabled_features)
            }


class ProductionHardeningManager:
    """Main orchestrator for production hardening."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.failover_managers: Dict[str, FailoverManager] = {}
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.bulkheads: Dict[str, BulkheadPattern] = {}
        self.health_checker = HealthChecker()
        self.graceful_degradation = GracefulDegradation()
        
        # Register default features
        self._register_default_features()
        
        self.lock = Lock()
    
    def _register_default_features(self):
        """Register default system features."""
        features = [
            ("predictions", []),
            ("explanations", ["predictions"]),
            ("reporting", ["predictions"]),
            ("data_export", ["predictions"]),
        ]
        
        for name, deps in features:
            self.graceful_degradation.register_feature(name, deps)
    
    def add_circuit_breaker(self, name: str, config: CircuitBreakerConfig = None):
        """Add circuit breaker."""
        if config is None:
            config = CircuitBreakerConfig(name=name)
        
        self.circuit_breakers[name] = CircuitBreaker(config)
    
    def add_rate_limiter(self, name: str, rate: int, per_seconds: int = 1):
        """Add rate limiter."""
        self.rate_limiters[name] = RateLimiter(rate, per_seconds)
    
    def add_bulkhead(self, name: str, max_concurrent: int = 10):
        """Add bulkhead."""
        self.bulkheads[name] = BulkheadPattern(name, max_concurrent)
    
    def register_health_check(self, name: str, check_func: Callable):
        """Register health check."""
        self.health_checker.register_check(name, check_func)
    
    def get_resilience_dashboard(self) -> Dict:
        """Get comprehensive resilience dashboard."""
        dashboard = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_health": self.health_checker.get_overall_health(),
            "circuit_breakers": {
                name: cb.get_status() for name, cb in self.circuit_breakers.items()
            },
            "rate_limiters": {
                name: rl.get_stats() for name, rl in self.rate_limiters.items()
            },
            "bulkheads": {
                name: bh.get_status() for name, bh in self.bulkheads.items()
            },
            "graceful_degradation": self.graceful_degradation.get_status(),
            "health_checks": self.health_checker.get_health_status()
        }
        
        return dashboard
    
    def handle_service_degradation(self, reason: str):
        """Gracefully degrade services during failures."""
        logger.warning(f"Initiating graceful degradation: {reason}")
        
        # Disable non-critical features
        self.graceful_degradation.disable_feature("explanations", reason)
        self.graceful_degradation.disable_feature("reporting", reason)
        
        # Keep core predictions available
        self.graceful_degradation.enable_feature("predictions")
    
    def recover_services(self):
        """Recover disabled services."""
        logger.info("Attempting service recovery...")
        
        # Re-enable features
        self.graceful_degradation.enable_feature("explanations")
        self.graceful_degradation.enable_feature("reporting")
        
        logger.info("Services recovered")


# Global instance
production_hardening_manager = ProductionHardeningManager()

# Initialize default components
production_hardening_manager.add_circuit_breaker("predictions", 
    CircuitBreakerConfig(failure_threshold=5, success_threshold=2, timeout_secs=60, name="predictions"))
production_hardening_manager.add_circuit_breaker("database",
    CircuitBreakerConfig(failure_threshold=3, success_threshold=2, timeout_secs=30, name="database"))

production_hardening_manager.add_rate_limiter("predictions", rate=100, per_seconds=1)
production_hardening_manager.add_rate_limiter("bulk_uploads", rate=10, per_seconds=60)

production_hardening_manager.add_bulkhead("prediction_workers", max_concurrent=20)
production_hardening_manager.add_bulkhead("database_connections", max_concurrent=10)
