import os
import shutil

import numpy as np
import pytest

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model import NormativeModel
from pcntoolkit.regression_model.test_model import TestModel
from test.fixtures.norm_data_fixtures import *
from test.fixtures.path_fixtures import *


@pytest.fixture
def test_model_args(save_dir_test_model):
    return {
        "savemodel": False,
        "saveresults": False,
        "evaluate_model": False,
        "saveplots": False,
        "save_dir": save_dir_test_model,
        "inscaler": "standardize",
        "outscaler": "standardize",
        "name": "test_model",
        "alg": "test_model",
        "success_ratio": 1.0,
    }


@pytest.fixture
def test_model():
    return TestModel("test_model")


@pytest.fixture
def new_norm_test_model(test_model, save_dir_test_model):
    if os.path.exists(save_dir_test_model):
        shutil.rmtree(save_dir_test_model)
    os.makedirs(save_dir_test_model, exist_ok=True)
    return NormativeModel(test_model, save_dir=save_dir_test_model)


@pytest.fixture
def fitted_norm_test_model(new_norm_test_model: NormativeModel, norm_data_from_arrays: NormData):
    if os.path.exists(new_norm_test_model.save_dir):
        shutil.rmtree(new_norm_test_model.save_dir)
    os.makedirs(new_norm_test_model.save_dir, exist_ok=True)
    new_norm_test_model.fit(norm_data_from_arrays)
    return new_norm_test_model

@pytest.fixture
def positive_norm_data(
    train_arrays: tuple,
) -> NormData:
    """Create NormData with strictly positive Y values.

    Takes the standard training arrays and shifts Y so
    that every element is strictly greater than zero.

    Returns
    -------
    NormData
        Dataset whose Y column values are all > 0.
    """
    # Unpack the standard training arrays
    X, y, batch_effects = train_arrays
    # Shift Y so every value is strictly positive
    y_positive = np.abs(y) + 1.0
    # Build a NormData from the positive arrays
    return NormData.from_ndarrays(
        "positive_data", X, y_positive, batch_effects
    )


@pytest.fixture
def positive_test_norm_data(
    test_arrays: tuple,
) -> NormData:
    """Create test NormData with strictly positive Y values.

    Takes the standard test arrays and shifts Y so that
    every element is strictly greater than zero.

    Returns
    -------
    NormData
        Dataset whose Y column values are all > 0.
    """
    # Unpack the standard test arrays
    X, y, batch_effects = test_arrays
    # Shift Y so every value is strictly positive
    y_positive = np.abs(y) + 1.0
    # Build a NormData from the positive arrays
    return NormData.from_ndarrays(
        "positive_test_data", X, y_positive, batch_effects
    )


@pytest.fixture
def norm_test_model_with_log_transform(
    save_dir_test_model: str,
) -> NormativeModel:
    """Create a NormativeModel using TestModel with log1p.

    The model has ``y_transform='log1p'`` so that Y is
    log-transformed before fitting and back-transformed
    after prediction, ensuring all outputs remain positive.

    Returns
    -------
    NormativeModel
        Un-fitted normative model with log1p transform.
    """
    # Build a fresh save directory
    log_dir = os.path.join(save_dir_test_model, "log1p")
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    # Create the underlying test regression model
    test_model = TestModel("test_model_log1p")
    # Return a NormativeModel with the log1p transform
    return NormativeModel(
        template_regression_model=test_model,
        savemodel=False,
        saveresults=False,
        evaluate_model=False,
        saveplots=False,
        save_dir=log_dir,
        inscaler="standardize",
        outscaler="standardize",
        y_transform="log1p",
        name="test_model_log1p",
    )
