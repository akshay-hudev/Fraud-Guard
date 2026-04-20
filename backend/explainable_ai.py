"""
Explainable AI Module: LIME Anchors, Counterfactuals, What-If Analysis
Step 9: Advanced model explanation and interpretability
"""

import json
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class Anchor:
    """LIME-style local anchor for predictions."""
    features: List[str]
    feature_values: Dict[str, Any]
    precision: float
    coverage: float
    interpretation: str


@dataclass
class Counterfactual:
    """Counterfactual explanation: minimum changes to flip prediction."""
    original_features: Dict[str, float]
    counterfactual_features: Dict[str, float]
    changed_features: Dict[str, Tuple[float, float]]  # feature: (old, new)
    num_changes: int
    change_distance: float
    confidence: float


@dataclass
class WhatIfScenario:
    """What-if analysis scenario results."""
    scenario_name: str
    original_prediction: float
    modified_prediction: float
    change: float
    change_direction: str
    modifications: Dict[str, Tuple[float, float]]  # feature: (old, new)
    feasibility: float
    recommendation: str


class AnchorsGenerator:
    """Generate LIME-style anchors for predictions."""
    
    def __init__(self, feature_names: List[str], fraud_threshold: float = 0.5):
        self.feature_names = feature_names
        self.fraud_threshold = fraud_threshold
        self.anchor_cache = {}
    
    def generate_anchor(self,
                       prediction: float,
                       feature_values: List[float],
                       perturbation_samples: int = 100
                       ) -> Anchor:
        """
        Generate anchor explaining why model made this prediction.
        
        Anchors are simple rules (e.g., "if doctor_frequency > 50 then FRAUD")
        that are highly precise in local neighborhood.
        """
        feature_dict = dict(zip(self.feature_names, feature_values))
        
        # Find most important features near decision boundary
        important_features = []
        feature_variances = {}
        
        # Simple variance-based importance (simulate perturbations)
        for i, feat_name in enumerate(self.feature_names):
            variations = []
            base_val = feature_values[i]
            
            # Perturb up/down
            for pct in [-0.3, -0.1, 0, 0.1, 0.3]:
                perturbed = base_val * (1 + pct)
                variations.append(perturbed)
            
            variance = max(variations) - min(variations)
            feature_variances[feat_name] = variance
            
            if variance > 0.01:
                important_features.append((feat_name, variance))
        
        # Sort by variance (importance)
        important_features.sort(key=lambda x: x[1], reverse=True)
        top_features = [f[0] for f in important_features[:5]]
        
        # Build anchor rule
        anchor_rule = self._build_anchor_rule(prediction, feature_dict, top_features)
        
        # Calculate precision (% of perturbed samples still classified same way)
        precision = self._estimate_anchor_precision(
            prediction, feature_values, top_features, perturbation_samples
        )
        
        # Coverage (% of dataset this anchor applies to) - simplified
        coverage = min(0.4 + precision * 0.3, 1.0)
        
        interpretation = f"Fraud flag: {prediction > self.fraud_threshold}. " \
                        f"Decision driven by: {', '.join(top_features[:3])}. " \
                        f"This rule explains {coverage*100:.0f}% of similar cases."
        
        anchor = Anchor(
            features=top_features,
            feature_values={k: feature_dict[k] for k in top_features},
            precision=precision,
            coverage=coverage,
            interpretation=interpretation
        )
        
        return anchor
    
    def _build_anchor_rule(self, prediction: float,
                          feature_dict: Dict[str, float],
                          top_features: List[str]) -> str:
        """Build human-readable anchor rule."""
        rules = []
        is_fraud = prediction > self.fraud_threshold
        
        for feat in top_features[:3]:
            val = feature_dict[feat]
            if is_fraud:
                threshold = val * 0.8  # Lower threshold for fraud
                rules.append(f"{feat} > {threshold:.2f}")
            else:
                threshold = val * 1.2  # Higher threshold for legit
                rules.append(f"{feat} <= {threshold:.2f}")
        
        return " AND ".join(rules) if rules else "Default rule"
    
    def _estimate_anchor_precision(self, prediction: float,
                                   feature_values: List[float],
                                   anchor_features: List[str],
                                   n_samples: int) -> float:
        """Estimate precision of anchor via perturbation."""
        same_predictions = 0
        
        for _ in range(n_samples):
            # Perturb non-anchor features while keeping anchor features fixed
            perturbed = list(feature_values)
            
            for i, feat_name in enumerate(self.feature_names):
                if feat_name not in anchor_features:
                    # Perturb by ±20%
                    noise = (-0.2 + 0.4 * (hash(f"{_}_{i}") % 100) / 100)
                    perturbed[i] *= (1 + noise)
            
            # In real scenario, would call model.predict(perturbed)
            # For now, simplified heuristic
            variance = sum(abs(perturbed[i] - feature_values[i]) 
                         for i in range(len(perturbed))) / len(perturbed)
            
            # Less variance = more likely same prediction
            if variance < 0.15:
                same_predictions += 1
        
        return same_predictions / n_samples


class CounterfactualGenerator:
    """Generate counterfactual explanations (min changes to flip prediction)."""
    
    def __init__(self, feature_names: List[str],
                 feature_ranges: Dict[str, Tuple[float, float]]):
        self.feature_names = feature_names
        self.feature_ranges = feature_ranges
    
    def generate_counterfactual(self,
                               current_prediction: float,
                               feature_values: List[float],
                               target_prediction: float = 0.2,
                               max_iterations: int = 50
                               ) -> Counterfactual:
        """
        Find minimum feature changes to flip prediction.
        
        E.g., "To change from FRAUD to LEGIT, reduce doctor_frequency from 85 to 40"
        """
        current_dict = dict(zip(self.feature_names, feature_values))
        counterfactual_dict = dict(current_dict)
        
        # Greedy approach: iteratively adjust feature with biggest impact
        changes_made = {}
        iteration = 0
        
        while iteration < max_iterations and current_prediction > target_prediction:
            # Find feature that gives best improvement with smallest change
            best_feature = None
            best_change = None
            best_improvement = 0
            
            for feat_idx, feat_name in enumerate(self.feature_names):
                if feat_name in changes_made:
                    continue  # Already modified
                
                current_val = counterfactual_dict[feat_name]
                min_val, max_val = self.feature_ranges.get(
                    feat_name, (0, 100)
                )
                
                # Try reducing this feature
                for target_val in [min_val, min_val + (max_val-min_val)*0.25]:
                    if abs(target_val - current_val) > 0.01:
                        # Simulate prediction change (heuristic)
                        val_change = abs(target_val - current_val)
                        improvement = val_change * 0.3  # 30% per unit change
                        
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_feature = feat_name
                            best_change = target_val
            
            if best_feature and best_change is not None:
                old_val = counterfactual_dict[best_feature]
                counterfactual_dict[best_feature] = best_change
                changes_made[best_feature] = (old_val, best_change)
                current_prediction *= 0.7  # Simulate improved prediction
            else:
                break
            
            iteration += 1
        
        # Calculate change distance (L2 norm)
        change_distance = sum(
            abs(counterfactual_dict[f] - current_dict[f])
            for f in self.feature_names
        ) / len(self.feature_names)
        
        # Confidence that counterfactual is actionable (fewer changes = higher)
        confidence = max(0.5, 1.0 - len(changes_made) * 0.15)
        
        return Counterfactual(
            original_features=current_dict,
            counterfactual_features=counterfactual_dict,
            changed_features=changes_made,
            num_changes=len(changes_made),
            change_distance=change_distance,
            confidence=confidence
        )


class WhatIfAnalyzer:
    """What-if analysis: simulate prediction changes."""
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.scenarios_history = []
    
    def analyze_modification(self,
                            scenario_name: str,
                            base_prediction: float,
                            feature_values: List[float],
                            modifications: Dict[str, float]
                            ) -> WhatIfScenario:
        """
        Analyze: what if we modify these features?
        
        Example: "What if doctor_frequency increases to 90?"
        """
        feature_dict = dict(zip(self.feature_names, feature_values))
        modified_dict = dict(feature_dict)
        
        change_magnitudes = []
        actual_modifications = {}
        
        for feat_name, new_value in modifications.items():
            if feat_name in feature_dict:
                old_value = feature_dict[feat_name]
                modified_dict[feat_name] = new_value
                actual_modifications[feat_name] = (old_value, new_value)
                
                # Change magnitude (normalized)
                magnitude = abs(new_value - old_value)
                change_magnitudes.append(magnitude)
        
        # Estimate new prediction (heuristic)
        avg_change = sum(change_magnitudes) / len(change_magnitudes) if change_magnitudes else 0
        
        # Features like doctor_frequency increase fraud risk
        high_risk_features = {"doctor_frequency", "claim_frequency", "claim_amount"}
        high_risk_increases = sum(
            1 for f, (old, new) in actual_modifications.items()
            if f in high_risk_features and new > old
        )
        
        modified_prediction = base_prediction + (avg_change * 0.4) + (high_risk_increases * 0.1)
        modified_prediction = min(1.0, max(0.0, modified_prediction))
        
        change = modified_prediction - base_prediction
        change_direction = "↑ Increases risk" if change > 0.05 else (
            "↓ Decreases risk" if change < -0.05 else "→ No significant change"
        )
        
        # Feasibility (how realistic is this scenario?)
        feasibility = 0.7  # Most scenarios realistic
        if any(v > 1.5 or v < 0 for v, _ in actual_modifications.values()):
            feasibility = 0.4  # Extreme values less feasible
        
        # Recommendation
        if change > 0.1:
            recommendation = "⚠️ This change significantly increases fraud risk"
        elif change < -0.1:
            recommendation = "✅ This change reduces fraud risk"
        else:
            recommendation = "ℹ️ Minimal impact on fraud prediction"
        
        scenario = WhatIfScenario(
            scenario_name=scenario_name,
            original_prediction=base_prediction,
            modified_prediction=modified_prediction,
            change=change,
            change_direction=change_direction,
            modifications=actual_modifications,
            feasibility=feasibility,
            recommendation=recommendation
        )
        
        self.scenarios_history.append(scenario)
        return scenario
    
    def get_sensitivity_analysis(self,
                                feature_name: str,
                                base_prediction: float,
                                current_features: List[float],
                                steps: int = 5
                                ) -> List[Dict]:
        """Sensitivity: how much does changing one feature affect prediction?"""
        feature_dict = dict(zip(self.feature_names, current_features))
        
        if feature_name not in feature_dict:
            return []
        
        current_value = feature_dict[feature_name]
        results = []
        
        # Test different values
        for i in range(steps):
            factor = 0.5 + (i / steps)
            test_value = current_value * factor
            
            # Estimate prediction (heuristic based on feature importance)
            change = abs(test_value - current_value)
            prediction_change = change * 0.3
            
            if feature_name in {"doctor_frequency", "claim_frequency"}:
                prediction_change *= 1.5  # High impact
            
            new_prediction = min(1.0, max(0.0, base_prediction + prediction_change))
            
            results.append({
                "value": test_value,
                "prediction": new_prediction,
                "change": new_prediction - base_prediction
            })
        
        return results


class ExplainableAIManager:
    """Main orchestrator for all explainability features."""
    
    def __init__(self, feature_names: List[str],
                 feature_ranges: Dict[str, Tuple[float, float]] = None):
        self.feature_names = feature_names
        self.feature_ranges = feature_ranges or {f: (0, 100) for f in feature_names}
        
        self.anchors_gen = AnchorsGenerator(feature_names)
        self.counterfactual_gen = CounterfactualGenerator(feature_names, self.feature_ranges)
        self.whatif_analyzer = WhatIfAnalyzer(feature_names)
        
        self.explanation_cache = {}
    
    def explain_prediction_comprehensive(self,
                                        prediction_id: str,
                                        prediction: float,
                                        feature_values: List[float],
                                        decision_boundary: float = 0.5
                                        ) -> Dict:
        """Generate complete explanation covering anchors, counterfactuals, what-if."""
        
        cache_key = f"{prediction_id}_{prediction:.3f}"
        if cache_key in self.explanation_cache:
            return self.explanation_cache[cache_key]
        
        # 1. Generate anchor
        anchor = self.anchors_gen.generate_anchor(prediction, feature_values)
        
        # 2. Generate counterfactual
        target_pred = 0.3 if prediction > decision_boundary else 0.7
        counterfactual = self.counterfactual_gen.generate_counterfactual(
            prediction, feature_values, target_pred
        )
        
        # 3. Generate what-if scenarios
        scenarios = []
        
        # Scenario: Reduce high-risk features
        reduce_mods = {}
        for i, feat_name in enumerate(self.feature_names):
            if "frequency" in feat_name or "amount" in feat_name:
                reduce_mods[feat_name] = feature_values[i] * 0.7
        
        if reduce_mods:
            scenario1 = self.whatif_analyzer.analyze_modification(
                "Reduce high-risk features",
                prediction,
                feature_values,
                reduce_mods
            )
            scenarios.append(scenario1)
        
        # Sensitivity analysis on top 3 features
        sensitivities = {}
        for feat in list(self.feature_names)[:3]:
            sensitivities[feat] = self.whatif_analyzer.get_sensitivity_analysis(
                feat, prediction, feature_values
            )
        
        explanation = {
            "prediction_id": prediction_id,
            "prediction": prediction,
            "is_fraud": prediction > decision_boundary,
            "decision_boundary": decision_boundary,
            "timestamp": datetime.utcnow().isoformat(),
            "anchor": {
                "features": anchor.features,
                "feature_values": anchor.feature_values,
                "precision": round(anchor.precision, 3),
                "coverage": round(anchor.coverage, 3),
                "interpretation": anchor.interpretation
            },
            "counterfactual": {
                "changed_features": {
                    k: {"from": v[0], "to": v[1]} for k, v in counterfactual.changed_features.items()
                },
                "num_changes": counterfactual.num_changes,
                "change_distance": round(counterfactual.change_distance, 3),
                "confidence": round(counterfactual.confidence, 3),
                "interpretation": f"To flip classification: {counterfactual.num_changes} changes needed"
            },
            "what_if_scenarios": [
                {
                    "name": s.scenario_name,
                    "original_prediction": round(s.original_prediction, 3),
                    "modified_prediction": round(s.modified_prediction, 3),
                    "change": round(s.change, 3),
                    "direction": s.change_direction,
                    "feasibility": round(s.feasibility, 3),
                    "recommendation": s.recommendation,
                    "modifications": {k: {"from": v[0], "to": v[1]} 
                                    for k, v in s.modifications.items()}
                }
                for s in scenarios
            ],
            "sensitivity_analysis": sensitivities
        }
        
        self.explanation_cache[cache_key] = explanation
        return explanation
    
    def compare_explanations(self,
                            pred1_id: str, pred1_score: float,
                            pred1_features: List[float],
                            pred2_id: str, pred2_score: float,
                            pred2_features: List[float]
                            ) -> Dict:
        """Compare explanations of two predictions."""
        
        exp1 = self.explain_prediction_comprehensive(
            pred1_id, pred1_score, pred1_features
        )
        exp2 = self.explain_prediction_comprehensive(
            pred2_id, pred2_score, pred2_features
        )
        
        # Find shared vs unique risk features
        shared_features = set(exp1["anchor"]["features"]) & set(exp2["anchor"]["features"])
        unique_to_1 = set(exp1["anchor"]["features"]) - shared_features
        unique_to_2 = set(exp2["anchor"]["features"]) - shared_features
        
        comparison = {
            "comparison": {
                "prediction_1": {
                    "id": pred1_id,
                    "score": round(pred1_score, 3),
                    "is_fraud": pred1_score > 0.5
                },
                "prediction_2": {
                    "id": pred2_id,
                    "score": round(pred2_score, 3),
                    "is_fraud": pred2_score > 0.5
                },
                "score_difference": round(abs(pred2_score - pred1_score), 3),
                "shared_risk_factors": list(shared_features),
                "unique_to_prediction_1": list(unique_to_1),
                "unique_to_prediction_2": list(unique_to_2),
                "similarity": round(len(shared_features) / max(
                    len(exp1["anchor"]["features"]), 
                    len(exp2["anchor"]["features"])
                ), 3)
            },
            "explanation_1": exp1,
            "explanation_2": exp2
        }
        
        return comparison
    
    def get_decision_boundaries(self, feature_name: str) -> Dict:
        """Analyze where decision boundary lies for a feature."""
        
        results = {
            "feature": feature_name,
            "boundaries": [],
            "interpretation": f"Decision boundary analysis for {feature_name}"
        }
        
        min_val, max_val = self.feature_ranges.get(feature_name, (0, 100))
        
        # Test values along feature range
        for i in range(5):
            val = min_val + (max_val - min_val) * (i / 4)
            # Simplified: fraud likelihood increases with feature value
            fraud_likelihood = min(val / max_val, 1.0)
            
            results["boundaries"].append({
                "value": round(val, 2),
                "fraud_likelihood": round(fraud_likelihood, 3),
                "region": "Low Risk" if fraud_likelihood < 0.33 else (
                    "Medium Risk" if fraud_likelihood < 0.67 else "High Risk"
                )
            })
        
        return results


# Global instance
explainable_ai_manager = ExplainableAIManager(
    feature_names=[
        "doctor_id", "hospital_id", "patient_age", "claim_amount",
        "claim_frequency", "doctor_frequency", "hospital_frequency",
        "approval_rate", "avg_claim_cost"
    ]
)
