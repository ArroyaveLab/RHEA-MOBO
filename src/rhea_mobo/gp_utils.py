"""Gaussian-process model fitting and multi-objective acquisition-function construction."""

import torch
from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.utils.multi_objective.pareto import is_non_dominated
from gpytorch.mlls import ExactMarginalLogLikelihood


def fit_single_task_model(train_x: torch.Tensor, train_y: torch.Tensor) -> SingleTaskGP:
    """Fit a single-output GP to one objective's training data."""
    model = SingleTaskGP(train_x, train_y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def build_model(train_x: torch.Tensor, train_y: torch.Tensor) -> ModelListGP:
    """Fit one GP per objective column, dropping rows with any NaN objective value.

    Raises:
        RuntimeError: If every row has at least one NaN objective, leaving no
            usable training data.
    """
    mask = ~torch.isnan(train_y).any(dim=1)
    clean_x = train_x[mask]
    clean_y = train_y[mask]
    if clean_x.shape[0] == 0:
        raise RuntimeError("All training outputs are NaN.")
    models = [fit_single_task_model(clean_x, clean_y[:, i : i + 1]) for i in range(clean_y.shape[-1])]
    return ModelListGP(*models)


def get_acquisition(model: ModelListGP, train_y: torch.Tensor, ref_point: torch.Tensor) -> ExpectedHypervolumeImprovement:
    """Build an Expected Hypervolume Improvement acquisition function over the current Pareto front."""
    pareto_y = train_y[is_non_dominated(train_y)]
    partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=pareto_y)
    return ExpectedHypervolumeImprovement(model=model, ref_point=ref_point.tolist(), partitioning=partitioning)
