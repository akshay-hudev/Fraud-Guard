"""
Model Interpretability Module (Step 7)
- SHAP-like value approximation
- Partial dependence plot generation
- Feature interaction analysis
- Model explanation generation
- Prediction breakdown

Pure Python implementation - no external ML dependencies.
"""

from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import statistics


class ShapApproximator:
    """Approximate SHAP values using permutation importance."""
    
    def __init__(self, baseline_prediction: float = 0.5):
        self.baseline_prediction = baseline_prediction
        self.feature_importances = {}
    
    def calculate_shap_values(self, features: Dict[str, Any], 
                             feature_importance: Dict[str, float],
                             baseline: float = None) -> Dict[str, float]:
        """
        Approximate SHAP values for a prediction.
        
        SHAP values show how much each feature contributes to moving
        the prediction away from the baseline prediction.
        """
        baseline = baseline or self.baseline_prediction
        shap_values = {}
        
        # Normalize importances to sum to 1
        total_importance = sum(abs(v) for v in feature_importance.values()) or 1
        
        for feature, importance in feature_importance.items():
            # Scale importance by the prediction difference
            prediction_diff = 1.0 - baseline if baseline < 0.5 else baseline - 0
            scaled_importance = (importance / total_importance) * prediction_diff
            shap_values[feature] = round(scaled_importance, 4)
        
        return shap_values
    
    def get_contribution_breakdown(self, features: Dict[str, Any],
                                  feature_importance: Dict[str, float],
                                  prediction_score: float) -> List[Dict[str, Any]]:
        """Get feature contributions ranked by impact."""
        shap_values = self.calculate_shap_values(
            features, feature_importance, prediction_score
        )
        
        # Sort by absolute SHAP value
        ranked = sorted(
            shap_values.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        contributions = []
        cumulative = 0
        
        for feature, shap_val in ranked:
            cumulative += abs(shap_val)
            contributions.append({
                "feature": feature,
                "value": features.get(feature, "N/A"),
                "shap_value": shap_val,
                "cumulative_contribution": round(cumulative, 4),
                "direction": "increases fraud" if shap_val > 0 else "decreases fraud",
            })
        
        return contributions


class PartialDependencePlotter:
    """Generate partial dependence plots for features."""
    
    @staticmethod
    def estimate_partial_dependence(feature_name: str,
                                   feature_values: List[float],
                                   predictions: List[float]) -> List[Dict[str, Any]]:
        """
        Estimate partial dependence for a feature.
        
        Shows how the average prediction changes as a feature varies,
        holding other features constant.
        """
        if not feature_values or not predictions:
            return []
        
        # Create bins for the feature
        min_val = min(feature_values)
        max_val = max(feature_values)
        
        if min_val == max_val:
            return [{
                "feature_value": min_val,
                "avg_prediction": statistics.mean(predictions),
            }]
        
        n_bins = min(10, len(set(feature_values)))
        bin_width = (max_val - min_val) / n_bins
        
        pd_plot = []
        
        for i in range(n_bins):
            bin_start = min_val + i * bin_width
            bin_end = bin_start + bin_width
            
            # Get predictions in this bin
            bin_predictions = [
                p for f, p in zip(feature_values, predictions)
                if bin_start <= f < bin_end or (i == n_bins - 1 and f == max_val)
            ]
            
            if bin_predictions:
                avg_pred = statistics.mean(bin_predictions)
                pd_plot.append({
                    "feature_value": round((bin_start + bin_end) / 2, 2),
                    "avg_prediction": round(avg_pred, 4),
                    "sample_count": len(bin_predictions),
                })
        
        return pd_plot
    
    @staticmethod
    def get_feature_range_impact(feature_name: str,
                                feature_distribution: List[float],
                                predictions: List[float]) -> Dict[str, Any]:
        """Analyze impact of feature range on predictions."""
        if not feature_distribution or not predictions:
            return {}
        
        sorted_features = sorted(list(zip(feature_distribution, predictions)))
        
        # Low (0-25th percentile)
        q1_idx = len(sorted_features) // 4
        low_preds = [p for _, p in sorted_features[:q1_idx]]
        
        # High (75-100th percentile)
        q3_idx = 3 * len(sorted_features) // 4
        high_preds = [p for _, p in sorted_features[q3_idx:]]
        
        return {
            "feature": feature_name,
            "low_range_avg_prediction": round(statistics.mean(low_preds), 4) if low_preds else 0,
            "high_range_avg_prediction": round(statistics.mean(high_preds), 4) if high_preds else 0,
            "impact": round(
                (statistics.mean(high_preds) - statistics.mean(low_preds)) if (high_preds and low_preds) else 0,
                4
            ),
        }


class FeatureInteractionAnalyzer:
    """Analyze interactions between features."""
    
    @staticmethod
    def find_feature_interactions(features: Dict[str, Any],
                                 feature_importance: Dict[str, float],
                                 prediction_score: float) -> List[Dict[str, Any]]:
        """
        Identify feature interactions.
        
        Features interact if their combined effect is greater than the sum
        of individual effects.
        """
        interactions = []
        feature_list = list(feature_importance.items())
        
        # Analyze pairwise interactions (top features only)
        top_features = sorted(feature_list, key=lambda x: abs(x[1]), reverse=True)[:5]
        
        for i, (feat1, imp1) in enumerate(top_features):
            for feat2, imp2 in top_features[i+1:]:
                # Estimate interaction effect
                interaction_strength = (imp1 * imp2) / (max(abs(imp1), abs(imp2)) + 0.0001)
                
                if abs(interaction_strength) > 0.01:
                    interactions.append({
                        "feature1": feat1,
                        "feature2": feat2,
                        "interaction_strength": round(interaction_strength, 4),
                        "value1": features.get(feat1, "N/A"),
                        "value2": features.get(feat2, "N/A"),
                        "interpretation": f"{feat1} and {feat2} jointly influence fraud risk",
                    })
        
        return sorted(interactions, key=lambda x: abs(x["interaction_strength"]), reverse=True)
    
    @staticmethod
    def identify_interaction_patterns(feature_pairs: List[Tuple[str, str]],
                                     data_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify patterns in feature interactions."""
        patterns = {}
        
        for feat1, feat2 in feature_pairs:
            co_occurrences = 0
            high_risk_cooccurrences = 0
            
            for sample in data_samples:
                val1 = sample.get(feat1, None)
                val2 = sample.get(feat2, None)
                
                if val1 is not None and val2 is not None:
                    co_occurrences += 1
                    # Check if both features are high-risk values
                    if (isinstance(val1, (int, float)) and val1 > statistics.median([s.get(feat1, 0) for s in data_samples if isinstance(s.get(feat1), (int, float))])) and \
                       (isinstance(val2, (int, float)) and val2 > statistics.median([s.get(feat2, 0) for s in data_samples if isinstance(s.get(feat2), (int, float))])):
                        high_risk_cooccurrences += 1
            
            if co_occurrences > 0:
                pattern_key = f"{feat1}_{feat2}"
                patterns[pattern_key] = {
                    "cooccurrence_rate": round(high_risk_cooccurrences / co_occurrences, 2),
                }
        
        return patterns


class ModelExplainer:
    """Generate comprehensive model explanations."""
    
    def __init__(self):
        self.shap_approximator = ShapApproximator()
        self.pd_plotter = PartialDependencePlotter()
        self.interaction_analyzer = FeatureInteractionAnalyzer()
        self.explanation_history = []
    
    def explain_prediction(self,
                          prediction_id: str,
                          features: Dict[str, Any],
                          prediction_score: float,
                          feature_importance: Dict[str, float],
                          model_version: str = "unknown") -> Dict[str, Any]:
        """Generate comprehensive explanation for a prediction."""
        
        # Calculate SHAP values
        shap_values = self.shap_approximator.calculate_shap_values(
            features, feature_importance, prediction_score
        )
        
        # Get contribution breakdown
        contributions = self.shap_approximator.get_contribution_breakdown(
            features, feature_importance, prediction_score
        )
        
        # Find interactions
        interactions = self.interaction_analyzer.find_feature_interactions(
            features, feature_importance, prediction_score
        )
        
        # Generate textual explanation
        explanation_text = self._generate_explanation_text(
            prediction_score, contributions, interactions
        )
        
        explanation = {
            "prediction_id": prediction_id,
            "model_version": model_version,
            "prediction_score": round(prediction_score, 4),
            "prediction_label": "HIGH FRAUD RISK" if prediction_score > 0.7 else "MEDIUM FRAUD RISK" if prediction_score > 0.4 else "LOW FRAUD RISK",
            "confidence": round(abs(prediction_score - 0.5) * 2, 4),  # 0 to 1 scale
            "shap_values": shap_values,
            "contributions": contributions,
            "interactions": interactions[:5],  # Top 5 interactions
            "explanation": explanation_text,
            "generated_at": datetime.utcnow().isoformat(),
        }
        
        self.explanation_history.append(explanation)
        return explanation
    
    def _generate_explanation_text(self, prediction_score: float,
                                  contributions: List[Dict[str, Any]],
                                  interactions: List[Dict[str, Any]]) -> str:
        """Generate human-readable explanation."""
        parts = []
        
        # Risk level
        if prediction_score > 0.7:
            parts.append("This claim has a HIGH FRAUD RISK.")
        elif prediction_score > 0.4:
            parts.append("This claim has a MEDIUM FRAUD RISK.")
        else:
            parts.append("This claim has a LOW FRAUD RISK.")
        
        # Top contributing features
        if contributions:
            top_contrib = contributions[0]
            parts.append(f"The most influential factor is {top_contrib['feature']} which {top_contrib['direction']}.")
        
        # Top interaction
        if interactions:
            top_inter = interactions[0]
            parts.append(f"Notable interaction: {top_inter['feature1']} and {top_inter['feature2']} together amplify fraud signals.")
        
        # Confidence
        confidence = abs(prediction_score - 0.5) * 2
        if confidence > 0.8:
            parts.append("The model is highly confident in this assessment.")
        elif confidence > 0.5:
            parts.append("The model has moderate confidence in this assessment.")
        else:
            parts.append("The model's confidence is low; manual review recommended.")
        
        return " ".join(parts)
    
    def get_explanation(self, prediction_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a previously generated explanation."""
        for exp in reversed(self.explanation_history):
            if exp["prediction_id"] == prediction_id:
                return exp
        return None
    
    def compare_explanations(self, pred_id_1: str, pred_id_2: str) -> Dict[str, Any]:
        """Compare explanations of two predictions."""
        exp1 = self.get_explanation(pred_id_1)
        exp2 = self.get_explanation(pred_id_2)
        
        if not exp1 or not exp2:
            return {"error": "One or both explanations not found"}
        
        return {
            "prediction_1": {
                "id": pred_id_1,
                "score": exp1["prediction_score"],
                "label": exp1["prediction_label"],
            },
            "prediction_2": {
                "id": pred_id_2,
                "score": exp2["prediction_score"],
                "label": exp2["prediction_label"],
            },
            "score_difference": round(abs(exp1["prediction_score"] - exp2["prediction_score"]), 4),
            "similar_risk_factors": self._find_similar_factors(exp1, exp2),
            "different_risk_factors": self._find_different_factors(exp1, exp2),
        }
    
    def _find_similar_factors(self, exp1: Dict, exp2: Dict) -> List[str]:
        """Find common contributing factors."""
        factors1 = set(c["feature"] for c in exp1.get("contributions", [])[:3])
        factors2 = set(c["feature"] for c in exp2.get("contributions", [])[:3])
        return list(factors1 & factors2)
    
    def _find_different_factors(self, exp1: Dict, exp2: Dict) -> Dict[str, List[str]]:
        """Find differing factors."""
        factors1 = set(c["feature"] for c in exp1.get("contributions", [])[:3])
        factors2 = set(c["feature"] for c in exp2.get("contributions", [])[:3])
        
        return {
            "unique_to_1": list(factors1 - factors2),
            "unique_to_2": list(factors2 - factors1),
        }
    
    def get_interpretation_summary(self, n_explanations: int = 100) -> Dict[str, Any]:
        """Get summary of model interpretation across recent predictions."""
        recent = self.explanation_history[-n_explanations:]
        
        if not recent:
            return {"total_predictions": 0}
        
        # Calculate statistics
        scores = [e["prediction_score"] for e in recent]
        avg_score = statistics.mean(scores)
        
        # Count by risk level
        high_risk = sum(1 for e in recent if e["prediction_score"] > 0.7)
        medium_risk = sum(1 for e in recent if 0.4 <= e["prediction_score"] <= 0.7)
        low_risk = sum(1 for e in recent if e["prediction_score"] < 0.4)
        
        # Most impactful features
        feature_impact = {}
        for exp in recent:
            for contrib in exp.get("contributions", [])[:3]:
                feat = contrib["feature"]
                feature_impact[feat] = feature_impact.get(feat, 0) + abs(contrib["shap_value"])
        
        top_features = sorted(feature_impact.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_predictions": len(recent),
            "avg_fraud_score": round(avg_score, 4),
            "high_risk_count": high_risk,
            "medium_risk_count": medium_risk,
            "low_risk_count": low_risk,
            "most_impactful_features": [f[0] for f in top_features],
            "feature_impact_scores": dict(top_features),
        }


# Global instance
model_explainer = ModelExplainer()
