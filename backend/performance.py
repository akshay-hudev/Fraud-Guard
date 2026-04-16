"""
Performance Optimization Module (Step 6)
- In-memory response caching with TTL
- Batch inference optimization
- Model prediction caching
- Query optimization

No external dependencies - pure Python implementation.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import OrderedDict
import hashlib
import time


class CacheEntry:
    """Single cache entry with TTL."""
    
    def __init__(self, value: Any, ttl_seconds: int = 300):
        self.value = value
        self.created_at = time.time()
        self.ttl_seconds = ttl_seconds
        self.hit_count = 0
        self.last_accessed = self.created_at
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return (time.time() - self.created_at) > self.ttl_seconds
    
    def access(self) -> Any:
        """Get value and update access metadata."""
        self.hit_count += 1
        self.last_accessed = time.time()
        return self.value


class ResponseCache:
    """LRU cache for API responses with TTL."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def _make_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Create cache key from endpoint and params."""
        params_str = str(sorted(params.items()))
        key_str = f"{endpoint}:{params_str}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, endpoint: str, params: Dict[str, Any] = None) -> Optional[Any]:
        """Get value from cache if not expired."""
        if params is None:
            params = {}
        
        key = self._make_key(endpoint, params)
        
        if key not in self.cache:
            self.misses += 1
            return None
        
        entry = self.cache[key]
        if entry.is_expired():
            del self.cache[key]
            self.misses += 1
            return None
        
        # Move to end (LRU)
        self.cache.move_to_end(key)
        self.hits += 1
        return entry.access()
    
    def set(self, endpoint: str, params: Dict[str, Any], value: Any, ttl: int = None) -> None:
        """Store value in cache."""
        if params is None:
            params = {}
        
        key = self._make_key(endpoint, params)
        ttl = ttl or self.default_ttl
        
        self.cache[key] = CacheEntry(value, ttl)
        self.cache.move_to_end(key)
        
        # Remove oldest if over limit
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
    
    def clear(self) -> None:
        """Clear all cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 2),
            "utilization": round(len(self.cache) / self.max_size * 100, 2),
        }


class PredictionCache:
    """Cache for model predictions to avoid redundant computations."""
    
    def __init__(self, max_size: int = 5000, ttl_seconds: int = 600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
    
    def _hash_input(self, data: Dict[str, Any]) -> str:
        """Create hash of input data."""
        # Sort dict items for consistent hashing
        items = sorted((k, str(v)) for k, v in data.items())
        key_str = str(items)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached prediction."""
        key = self._hash_input(data)
        
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        if entry.is_expired():
            del self.cache[key]
            return None
        
        self.cache.move_to_end(key)
        return entry.access()
    
    def set(self, data: Dict[str, Any], prediction: Dict[str, Any]) -> None:
        """Cache a prediction."""
        key = self._hash_input(data)
        self.cache[key] = CacheEntry(prediction, self.ttl_seconds)
        self.cache.move_to_end(key)
        
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache stats."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
        }


class BatchOptimizer:
    """Optimize batch predictions through vectorization."""
    
    @staticmethod
    def group_by_size(records: List[Dict[str, Any]], batch_size: int = 32) -> List[List[Dict[str, Any]]]:
        """Group records into batches."""
        batches = []
        for i in range(0, len(records), batch_size):
            batches.append(records[i:i + batch_size])
        return batches
    
    @staticmethod
    def estimate_latency(batch_size: int, base_latency_ms: float = 50, 
                        per_record_ms: float = 5) -> float:
        """Estimate batch prediction latency."""
        return base_latency_ms + (batch_size * per_record_ms)
    
    @staticmethod
    def find_optimal_batch_size(total_records: int, max_latency_ms: int = 500,
                               base_latency_ms: float = 50,
                               per_record_ms: float = 5) -> int:
        """Find optimal batch size given constraints."""
        # Solve: base + (batch * per_record) <= max_latency
        optimal = int((max_latency_ms - base_latency_ms) / per_record_ms)
        optimal = max(1, min(optimal, total_records))
        return optimal


class QueryOptimizer:
    """Optimize database/feature queries."""
    
    def __init__(self):
        self.query_stats = {}
        self.slow_queries = []
    
    def track_query(self, query_name: str, duration_ms: float, threshold_ms: int = 100) -> None:
        """Track query execution time."""
        if query_name not in self.query_stats:
            self.query_stats[query_name] = {
                "count": 0,
                "total_time": 0,
                "avg_time": 0,
                "max_time": 0,
            }
        
        stats = self.query_stats[query_name]
        stats["count"] += 1
        stats["total_time"] += duration_ms
        stats["avg_time"] = stats["total_time"] / stats["count"]
        stats["max_time"] = max(stats["max_time"], duration_ms)
        
        if duration_ms > threshold_ms:
            self.slow_queries.append({
                "query": query_name,
                "duration_ms": duration_ms,
                "timestamp": datetime.utcnow().isoformat(),
            })
            self.slow_queries = self.slow_queries[-100:]  # Keep last 100
    
    def get_slow_queries(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get slowest queries."""
        return self.slow_queries[-limit:]
    
    def get_query_stats(self) -> Dict[str, Any]:
        """Get query statistics."""
        if not self.query_stats:
            return {}
        
        # Sort by avg time
        sorted_queries = sorted(
            self.query_stats.items(),
            key=lambda x: x[1]["avg_time"],
            reverse=True
        )
        
        return dict(sorted_queries[:10])  # Top 10 slowest


class PerformanceMonitor:
    """Main performance monitoring orchestrator."""
    
    def __init__(self):
        self.response_cache = ResponseCache(max_size=1000, default_ttl=300)
        self.prediction_cache = PredictionCache(max_size=5000, ttl_seconds=600)
        self.query_optimizer = QueryOptimizer()
        self.batch_optimizer = BatchOptimizer()
        
        self.request_times = []
        self.inference_times = []
    
    def track_request(self, endpoint: str, duration_ms: float) -> None:
        """Track API request latency."""
        self.request_times.append({
            "endpoint": endpoint,
            "duration_ms": duration_ms,
            "timestamp": time.time(),
        })
        # Keep last 1000
        self.request_times = self.request_times[-1000:]
    
    def track_inference(self, duration_ms: float) -> None:
        """Track model inference time."""
        self.inference_times.append(duration_ms)
        self.inference_times = self.inference_times[-1000:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary."""
        request_times = [r["duration_ms"] for r in self.request_times]
        
        if not request_times:
            return {
                "avg_request_time_ms": 0,
                "p95_request_time_ms": 0,
                "p99_request_time_ms": 0,
                "avg_inference_time_ms": 0,
            }
        
        sorted_requests = sorted(request_times)
        
        avg_request = sum(request_times) / len(request_times)
        p95_request = sorted_requests[int(len(sorted_requests) * 0.95)] if sorted_requests else 0
        p99_request = sorted_requests[int(len(sorted_requests) * 0.99)] if sorted_requests else 0
        
        avg_inference = sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0
        
        return {
            "avg_request_time_ms": round(avg_request, 2),
            "p95_request_time_ms": round(p95_request, 2),
            "p99_request_time_ms": round(p99_request, 2),
            "max_request_time_ms": round(max(request_times), 2) if request_times else 0,
            "avg_inference_time_ms": round(avg_inference, 2),
            "total_requests": len(self.request_times),
            "total_inferences": len(self.inference_times),
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get all cache statistics."""
        return {
            "response_cache": self.response_cache.get_stats(),
            "prediction_cache": self.prediction_cache.get_stats(),
        }
    
    def get_bottlenecks(self) -> Dict[str, Any]:
        """Identify performance bottlenecks."""
        summary = self.get_performance_summary()
        cache_stats = self.get_cache_stats()
        
        bottlenecks = []
        
        # Check request latency
        if summary["avg_request_time_ms"] > 200:
            bottlenecks.append({
                "type": "high_latency",
                "severity": "warning",
                "message": f"Average request time {summary['avg_request_time_ms']}ms exceeds 200ms threshold",
            })
        
        # Check inference time
        if summary["avg_inference_time_ms"] > 150:
            bottlenecks.append({
                "type": "slow_inference",
                "severity": "warning",
                "message": f"Average inference time {summary['avg_inference_time_ms']}ms is high",
            })
        
        # Check cache hit rate
        response_hit_rate = cache_stats["response_cache"]["hit_rate"]
        if response_hit_rate < 30:
            bottlenecks.append({
                "type": "low_cache_hits",
                "severity": "info",
                "message": f"Response cache hit rate {response_hit_rate}% is low - consider increasing TTL",
            })
        
        # Check p99 latency
        if summary["p99_request_time_ms"] > 500:
            bottlenecks.append({
                "type": "outlier_latency",
                "severity": "warning",
                "message": f"P99 latency {summary['p99_request_time_ms']}ms - some requests are very slow",
            })
        
        return {
            "bottlenecks": bottlenecks,
            "summary": summary,
        }


# Global instance
performance_monitor = PerformanceMonitor()
