"""
Test to verify that MSLL is computed correctly regardless of data scale.

This test addresses the issue where MSLL was scale-dependent because the baseline
log-likelihood was computed on unscaled Y values while the model log-likelihood
was computed on scaled Y values.

The fix ensures both are computed on the same (scaled) data.
"""

import numpy as np
import pytest
import xarray as xr

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model import NormativeModel
from pcntoolkit.regression_model.blr import BLR


def create_test_data(n_samples: int = 100, scale_factor: float = 1.0, seed: int = 42):
    """Create synthetic test data with a specified scale factor."""
    np.random.seed(seed)
    
    # Create covariates (age-like)
    X = np.random.uniform(20, 80, (n_samples, 1))
    
    # Create response variable with linear relationship + noise
    # Scale the response by scale_factor
    Y_base = 0.5 * X[:, 0] + np.random.normal(0, 5, n_samples)
    Y = (Y_base * scale_factor).reshape(-1, 1)
    
    # Create batch effects
    batch_effects = np.random.choice([0, 1], size=(n_samples, 1)).astype(str)
    
    # Create NormData
    data = NormData(
        name="test_data",
        data_vars={
            "X": xr.DataArray(X, dims=("observations", "covariates")),
            "Y": xr.DataArray(Y, dims=("observations", "response_vars")),
            "batch_effects": xr.DataArray(batch_effects, dims=("observations", "batch_effect_dims")),
            "subject_ids": xr.DataArray(np.arange(n_samples), dims=("observations",)),
        },
        coords={
            "observations": np.arange(n_samples),
            "covariates": ["age"],
            "response_vars": ["test_metric"],
            "batch_effect_dims": ["site"],
        },
    )
    
    return data


@pytest.mark.parametrize("scale_factor", [1.0, 1e-4, 1e-6, 1e3])
def test_msll_scale_independence(scale_factor, tmp_path):
    """
    Test that MSLL values are similar regardless of data scale.
    
    Previously, MSLL was artificially high for small-scale data because
    the baseline NLL was computed on unscaled Y while model NLL was on scaled Y.
    
    After the fix, MSLL should be comparable across different scales.
    """
    # Create data with specified scale
    data = create_test_data(n_samples=100, scale_factor=scale_factor, seed=42)
    
    # Create and fit model
    blr = BLR()
    model = NormativeModel(
        template_regression_model=blr,
        savemodel=False,
        evaluate_model=True,
        saveresults=False,
        saveplots=False,
        save_dir=str(tmp_path),
        inscaler="standardize",
        outscaler="standardize",
    )
    
    # Fit and predict
    model.fit(data)
    
    # Get MSLL
    msll = float(data.statistics.sel(response_vars="test_metric", statistic="MSLL").values)
    
    # MSLL should be reasonable (not extremely large positive values)
    # A well-fitted model should have MSLL close to 0 or negative
    # (meaning it's better than or similar to the baseline)
    print(f"Scale factor: {scale_factor}, MSLL: {msll}")
    
    # The key assertion: MSLL should not be extremely large
    # For a reasonably fitted model, MSLL should typically be < 2
    # (positive MSLL means worse than baseline, but shouldn't be huge)
    assert msll < 5.0, f"MSLL too large ({msll}) for scale_factor={scale_factor}"
    
    # Also check that baseline_logp was computed
    assert "baseline_logp" in data.data_vars, "baseline_logp should be stored in data"


def test_msll_consistency_across_scales(tmp_path):
    """
    Test that MSLL values are consistent when only the scale differs.
    
    Using the same random seed, data at different scales should produce
    similar MSLL values (within a reasonable tolerance).
    """
    msll_values = {}
    
    for scale_factor in [1.0, 1e-4]:
        # Create data with specified scale (same seed = same structure)
        data = create_test_data(n_samples=100, scale_factor=scale_factor, seed=123)
        
        # Create and fit model
        blr = BLR()
        model = NormativeModel(
            template_regression_model=blr,
            savemodel=False,
            evaluate_model=True,
            saveresults=False,
            saveplots=False,
            save_dir=str(tmp_path / f"scale_{scale_factor}"),
            inscaler="standardize",
            outscaler="standardize",
        )
        
        model.fit(data)
        
        msll = float(data.statistics.sel(response_vars="test_metric", statistic="MSLL").values)
        msll_values[scale_factor] = msll
        print(f"Scale factor: {scale_factor}, MSLL: {msll}")
    
    # The MSLL values should be similar regardless of original scale
    # (because both model and baseline are computed on standardized data)
    msll_diff = abs(msll_values[1.0] - msll_values[1e-4])
    print(f"MSLL difference between scales: {msll_diff}")
    
    # Allow some numerical tolerance
    assert msll_diff < 0.5, (
        f"MSLL differs too much between scales: "
        f"scale=1.0 -> {msll_values[1.0]}, scale=1e-4 -> {msll_values[1e-4]}"
    )


if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        print("=" * 60)
        print("Testing MSLL scale independence...")
        print("=" * 60)
        
        for scale in [1.0, 1e-4, 1e-6]:
            test_msll_scale_independence(scale, tmp_path)
            print(f"✓ Scale {scale} passed")
        
        print("\n" + "=" * 60)
        print("Testing MSLL consistency across scales...")
        print("=" * 60)
        test_msll_consistency_across_scales(tmp_path)
        print("✓ Consistency test passed")
        
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
