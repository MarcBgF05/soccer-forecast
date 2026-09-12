"""Dixon-Coles model and ADK-ready prediction tools for La Liga."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.optimize import minimize
from scipy.stats import poisson


XI = 0.0018
DEFAULT_MAX_GOALS = 12
REQUIRED_COLUMNS = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _rho_correction(x: int, y: int, lambda_x: float, mu_y: float, rho: float) -> float:
    """Return the Dixon-Coles low-score correction (tau)."""
    if x == 0 and y == 0:
        return 1 - lambda_x * mu_y * rho
    if x == 0 and y == 1:
        return 1 + lambda_x * rho
    if x == 1 and y == 0:
        return 1 + mu_y * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def _dc_log_likelihood(
    home_goals: int,
    away_goals: int,
    home_attack: float,
    home_defence: float,
    away_attack: float,
    away_defence: float,
    rho: float,
    home_advantage: float,
) -> float:
    home_mean = np.exp(home_attack + away_defence + home_advantage)
    away_mean = np.exp(away_attack + home_defence)
    tau = max(_rho_correction(home_goals, away_goals, home_mean, away_mean, rho), 1e-10)
    return (
        np.log(tau)
        + np.log(max(poisson.pmf(home_goals, home_mean), 1e-10))
        + np.log(max(poisson.pmf(away_goals, away_mean), 1e-10))
    )


def _data_paths() -> tuple[Path, Path]:
    """Read configured CSV paths, falling back to this repository's datasets."""
    load_dotenv()
    first_season = Path(os.getenv("SP_2025_2026", PROJECT_ROOT / "datasets" / "sp_2526.csv"))
    second_season = Path(os.getenv("SP_2026_2027", PROJECT_ROOT / "datasets" / "sp_2627.csv"))
    missing = [str(path) for path in (first_season, second_season) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"No se encontraron los archivos de partidos: {', '.join(missing)}")
    return first_season, second_season


def _load_match_data() -> pd.DataFrame:
    """Load, clean and time-weight the configured historical results."""
    frames = []
    for path in _data_paths():
        frame = pd.read_csv(path)
        missing_columns = REQUIRED_COLUMNS.difference(frame.columns)
        if missing_columns:
            raise ValueError(f"{path} no contiene las columnas requeridas: {sorted(missing_columns)}")
        frames.append(frame[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]])

    data = pd.concat(frames, ignore_index=True).rename(
        columns={"FTHG": "HomeGoals", "FTAG": "AwayGoals"}
    )
    data["Date"] = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")
    data["HomeGoals"] = pd.to_numeric(data["HomeGoals"], errors="coerce")
    data["AwayGoals"] = pd.to_numeric(data["AwayGoals"], errors="coerce")
    data = data.dropna(subset=["Date", "HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals"])
    data = data.sort_values("Date").reset_index(drop=True)
    data["HomeGoals"] = data["HomeGoals"].astype(int)
    data["AwayGoals"] = data["AwayGoals"].astype(int)
    data["time_diff"] = (pd.Timestamp.now().normalize() - data["Date"]).dt.days.clip(lower=0)
    return data


def _fit_parameters(dataset: pd.DataFrame, xi: float = XI) -> dict[str, float]:
    """Fit Dixon-Coles attack, defence, correlation and home-advantage parameters."""
    team_names = np.sort(list(set(dataset["HomeTeam"]) | set(dataset["AwayTeam"])))
    number_of_teams = len(team_names)
    if number_of_teams < 2:
        raise ValueError("Se necesitan resultados de al menos dos equipos para entrenar el modelo.")

    rng = np.random.default_rng(42)
    initial_values = np.concatenate(
        (
            rng.uniform(0, 1, number_of_teams),
            rng.uniform(-1, 0, number_of_teams),
            np.array([0.0, 0.3]),
        )
    )
    home_indices = np.searchsorted(team_names, dataset["HomeTeam"].to_numpy())
    away_indices = np.searchsorted(team_names, dataset["AwayTeam"].to_numpy())
    home_goals = dataset["HomeGoals"].to_numpy(dtype=float)
    away_goals = dataset["AwayGoals"].to_numpy(dtype=float)
    weights = np.exp(-xi * dataset["time_diff"].to_numpy(dtype=float))

    def negative_log_likelihood(parameters: np.ndarray) -> float:
        attack = parameters[:number_of_teams]
        defence = parameters[number_of_teams : 2 * number_of_teams]
        rho, home_advantage = parameters[-2:]
        home_means = np.exp(attack[home_indices] + defence[away_indices] + home_advantage)
        away_means = np.exp(attack[away_indices] + defence[home_indices])
        tau = np.ones(len(dataset))
        zero_zero = (home_goals == 0) & (away_goals == 0)
        zero_one = (home_goals == 0) & (away_goals == 1)
        one_zero = (home_goals == 1) & (away_goals == 0)
        one_one = (home_goals == 1) & (away_goals == 1)
        tau[zero_zero] = 1 - home_means[zero_zero] * away_means[zero_zero] * rho
        tau[zero_one] = 1 + home_means[zero_one] * rho
        tau[one_zero] = 1 + away_means[one_zero] * rho
        tau[one_one] = 1 - rho
        log_likelihood = np.log(np.maximum(tau, 1e-10))
        log_likelihood += poisson.logpmf(home_goals, home_means) + poisson.logpmf(away_goals, away_means)
        return float(-np.sum(weights * log_likelihood))

    constraint = {"type": "eq", "fun": lambda values: sum(values[:number_of_teams]) - number_of_teams}
    result = minimize(
        negative_log_likelihood,
        initial_values,
        method="SLSQP",
        constraints=[constraint],
        options={"disp": False, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(f"No fue posible ajustar el modelo Dixon-Coles: {result.message}")

    parameter_names = (
        [f"attack_{team}" for team in team_names]
        + [f"defence_{team}" for team in team_names]
        + ["rho", "home_adv"]
    )
    return dict(zip(parameter_names, result.x))


@lru_cache(maxsize=1)
def _model_parameters() -> dict[str, float]:
    """Train once per process; both ADK tools reuse the resulting parameters."""
    return _fit_parameters(_load_match_data())


def _validate_teams(parameters: dict[str, float], home_team: str, away_team: str) -> tuple[str, str]:
    home_team, away_team = home_team.strip(), away_team.strip()
    if not home_team or not away_team:
        raise ValueError("Debes indicar un equipo local y un equipo visitante.")
    available_teams = {key.removeprefix("attack_") for key in parameters if key.startswith("attack_")}
    unknown = sorted({team for team in (home_team, away_team) if team not in available_teams})
    if unknown:
        raise ValueError(
            f"Equipo(s) no disponible(s) en los datos históricos: {', '.join(unknown)}. "
            f"Equipos disponibles: {', '.join(sorted(available_teams))}."
        )
    return home_team, away_team


def _goal_means(parameters: dict[str, float], home_team: str, away_team: str) -> tuple[float, float]:
    home_mean = np.exp(
        parameters[f"attack_{home_team}"]
        + parameters[f"defence_{away_team}"]
        + parameters["home_adv"]
    )
    away_mean = np.exp(parameters[f"attack_{away_team}"] + parameters[f"defence_{home_team}"])
    return float(home_mean), float(away_mean)


def _score_probability(
    parameters: dict[str, float], home_team: str, away_team: str, home_goals: int, away_goals: int
) -> float:
    home_mean, away_mean = _goal_means(parameters, home_team, away_team)
    tau = _rho_correction(home_goals, away_goals, home_mean, away_mean, parameters["rho"])
    return float(max(tau, 0.0) * poisson.pmf(home_goals, home_mean) * poisson.pmf(away_goals, away_mean))


def predict_match_outcome(home_team: str, away_team: str) -> dict[str, Any]:
    """Predice victoria local, empate y victoria visitante para un partido de La Liga.

    Args:
        home_team: Nombre del equipo que juega como local, tal como aparece en los CSV.
        away_team: Nombre del equipo que juega como visitante, tal como aparece en los CSV.
    """
    parameters = _model_parameters()
    home_team, away_team = _validate_teams(parameters, home_team, away_team)
    probabilities = np.array(
        [
            [_score_probability(parameters, home_team, away_team, home_goals, away_goals)
             for away_goals in range(DEFAULT_MAX_GOALS + 1)]
            for home_goals in range(DEFAULT_MAX_GOALS + 1)
        ]
    )
    home_win = float(np.tril(probabilities, -1).sum())
    draw = float(np.trace(probabilities))
    away_win = float(np.triu(probabilities, 1).sum())
    home_mean, away_mean = _goal_means(parameters, home_team, away_team)
    return {
        "local": home_team,
        "visitante": away_team,
        "goles_esperados_local": round(home_mean, 3),
        "goles_esperados_visitante": round(away_mean, 3),
        "probabilidad_victoria_local": round(home_win * 100, 2),
        "probabilidad_empate": round(draw * 100, 2),
        "probabilidad_victoria_visitante": round(away_win * 100, 2),
    }


def predict_exact_score(home_team: str, away_team: str, home_goals: int, away_goals: int) -> dict[str, Any]:
    """Calcula la probabilidad de un marcador exacto para un partido de La Liga.

    Args:
        home_team: Nombre del equipo que juega como local.
        away_team: Nombre del equipo que juega como visitante.
        home_goals: Cantidad de goles del local en el marcador consultado, cero o mayor.
        away_goals: Cantidad de goles del visitante en el marcador consultado, cero o mayor.
    """
    if (
        isinstance(home_goals, bool)
        or isinstance(away_goals, bool)
        or not isinstance(home_goals, (int, np.integer))
        or not isinstance(away_goals, (int, np.integer))
        or home_goals < 0
        or away_goals < 0
    ):
        raise ValueError("Los goles deben ser números enteros iguales o mayores que cero.")
    parameters = _model_parameters()
    home_team, away_team = _validate_teams(parameters, home_team, away_team)
    probability = _score_probability(parameters, home_team, away_team, home_goals, away_goals)
    return {
        "local": home_team,
        "visitante": away_team,
        "marcador": f"{home_team} {home_goals}-{away_goals} {away_team}",
        "probabilidad": round(probability * 100, 2),
    }
