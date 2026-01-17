# SHAP (SHapley Additive exPlanations) Integration Guide

## Overview

SHAP provides model interpretability for production trading systems. It explains individual predictions and provides global feature importance with directionality.

## Why SHAP?

- **Regulatory compliance**: Required for explainable AI in financial systems
- **Debugging**: Understand why models make specific predictions
- **Feature engineering**: Identify which features matter most
- **Directionality**: Understand positive vs negative feature impacts
- **Interaction detection**: Find feature interactions

## Key Features

### 1. TreeExplainer for XGBoost

**Purpose**: Fast SHAP value calculation for tree-based models

**Usage**:
```python
import shap
import xgboost as xgb

# Train model
model = xgb.XGBRegressor()
model.fit(X_train, y_train)

# Create explainer
explainer = shap.TreeExplainer(model)

# Calculate SHAP values
shap_values = explainer.shap_values(X_test)
```

**Benefits**:
- Fast for tree models (polynomial time)
- Exact SHAP values (not approximations)
- Works with XGBoost GPU models

### 2. Local Explanations (Individual Predictions)

**Purpose**: Explain why a specific prediction was made

**Usage**:
```python
# Explain single prediction
shap_values_single = explainer.shap_values(X_test.iloc[0:1])

# Waterfall plot
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values_single[0],
        base_values=explainer.expected_value,
        data=X_test.iloc[0],
        feature_names=X_test.columns
    )
)
```

**Benefits**:
- Understand individual trade decisions
- Regulatory compliance (explainable AI)
- Debugging specific predictions

### 3. Global Feature Importance

**Purpose**: Understand overall feature importance with directionality

**Usage**:
```python
# Summary plot
shap.summary_plot(shap_values, X_test)

# Bar plot (mean absolute SHAP values)
shap.plots.bar(shap_values)

# Feature importance with direction
shap.plots.beeswarm(shap_values)
```

**Benefits**:
- Identify most important features
- Understand positive vs negative impacts
- Feature selection guidance

### 4. Feature Interaction Detection

**Purpose**: Find interactions between features

**Usage**:
```python
# Interaction values
shap_interaction_values = explainer.shap_interaction_values(X_test)

# Interaction plot
shap.summary_plot(
    shap_interaction_values,
    X_test,
    plot_type="compact_dot"
)
```

**Benefits**:
- Discover feature interactions
- Improve feature engineering
- Understand complex model behavior

### 5. Regime-Dependent Importance

**Purpose**: Analyze feature importance in different market regimes

**Usage**:
```python
# Calculate SHAP values per regime
regimes = ["bullish", "bearish", "high_volatility"]
for regime in regimes:
    regime_mask = market_regime == regime
    regime_shap = shap_values[regime_mask]
    regime_X = X_test[regime_mask]

    shap.summary_plot(regime_shap, regime_X, title=f"{regime} Regime")
```

**Benefits**:
- Understand regime-specific feature importance
- Regime-aware feature selection
- Better model interpretation

## Integration Points

### In `src/nova/models/trainer.py`

Add SHAP calculation post-training:
```python
import shap

class ModelTrainer:
    def save_model(self, model, metadata: Dict[str, Any]) -> None:
        """Save model with SHAP analysis."""
        # Calculate SHAP values on validation set
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val)

        # Store SHAP summary statistics
        metadata["shap"] = {
            "mean_abs_shap": np.abs(shap_values).mean(axis=0).tolist(),
            "feature_names": feature_names,
            "expected_value": float(explainer.expected_value),
        }

        # Save model
        model.save_model(model_path)
```

### In `src/nova/models/predictor.py`

Add real-time explanation:
```python
import shap

class ModelPredictor:
    def __init__(self, model_path: str, explainer_path: Optional[str] = None):
        """Initialize predictor with optional SHAP explainer."""
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)

        if explainer_path:
            self.explainer = shap.TreeExplainer.load(explainer_path)
        else:
            self.explainer = shap.TreeExplainer(self.model)

    def explain_prediction(
        self,
        features: np.ndarray,
        return_dict: bool = True
    ) -> Dict[str, Any]:
        """
        Explain a single prediction.

        Returns:
            Dictionary with SHAP values and explanation
        """
        shap_values = self.explainer.shap_values(features)

        if return_dict:
            return {
                "prediction": float(self.model.predict(features)[0]),
                "base_value": float(self.explainer.expected_value),
                "shap_values": shap_values[0].tolist(),
                "feature_names": self.feature_names,
            }
        return shap_values
```

### In `src/nova/dashboard/app.py`

Add SHAP visualization:
```python
import shap
import streamlit as st
import plotly.graph_objects as go

def render_shap_analysis():
    """Render SHAP analysis in dashboard."""
    st.header("Model Interpretability (SHAP)")

    # Load SHAP values from model metadata
    shap_data = load_model_shap_metadata()

    # Summary plot
    st.subheader("Global Feature Importance")
    fig = create_shap_summary_plot(shap_data)
    st.plotly_chart(fig)

    # Individual prediction explanation
    st.subheader("Explain Individual Prediction")
    prediction_idx = st.selectbox("Select prediction", range(len(predictions)))

    shap_values_single = explainer.shap_values(X_test.iloc[prediction_idx:prediction_idx+1])

    # Waterfall plot
    fig = create_shap_waterfall_plot(shap_values_single[0])
    st.plotly_chart(fig)
```

## Storage in Database

Store SHAP values for recent predictions:

```python
# In predictor.py after prediction
shap_values = self.explain_prediction(features)

# Store in database
await db.execute("""
    INSERT INTO prediction_explanations (
        timestamp, symbol, prediction, shap_values, base_value
    ) VALUES ($1, $2, $3, $4, $5)
""", timestamp, symbol, prediction, shap_values, base_value)
```

## Visualization Functions

### Summary Plot (Plotly)
```python
def create_shap_summary_plot(shap_values: np.ndarray, feature_names: List[str]):
    """Create Plotly summary plot from SHAP values."""
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    fig = go.Figure(data=[
        go.Bar(
            x=mean_abs_shap,
            y=feature_names,
            orientation='h'
        )
    ])
    fig.update_layout(
        title="Feature Importance (Mean |SHAP|)",
        xaxis_title="Mean |SHAP Value|",
        yaxis_title="Feature"
    )
    return fig
```

### Waterfall Plot (Plotly)
```python
def create_shap_waterfall_plot(shap_values: np.ndarray, base_value: float):
    """Create Plotly waterfall plot."""
    # Sort by absolute SHAP value
    sorted_idx = np.argsort(np.abs(shap_values))[::-1]

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative"] * len(shap_values),
        x=feature_names[sorted_idx],
        textposition="outside",
        text=shap_values[sorted_idx],
        y=shap_values[sorted_idx],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))

    fig.update_layout(
        title="SHAP Waterfall Plot",
        showlegend=False
    )
    return fig
```

## Performance Considerations

- **TreeExplainer**: Fast for tree models (exact calculation)
- **Sampling**: For large datasets, sample SHAP calculation
- **Caching**: Cache explainer and SHAP values for repeated predictions
- **Batch processing**: Calculate SHAP values in batches

## Installation

```bash
pip install shap>=0.45.0
```

## References

- Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions"
- SHAP Documentation: https://shap.readthedocs.io/
- XGBoost SHAP: https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.plot_importance

## Impact

- **Regulatory compliance**: Full explainability for trading decisions
- **Debugging**: Understand model behavior
- **Feature engineering**: Identify important features
- **Transparency**: Explain individual predictions
