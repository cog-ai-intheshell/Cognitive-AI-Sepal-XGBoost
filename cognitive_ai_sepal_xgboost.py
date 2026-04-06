#!/usr/bin/env python3
"""
Pipeline "Cognitive AI" pour une classification binaire sur Iris.

Par defaut, le systeme n'utilise que les variables du sepale:
- sepal length (cm)
- sepal width (cm)

Le schema suit la logique:
Perception -> Etat interne eta' -> Projection P(y=1|x) -> Threshold tau
-> Action brute -> Reflexes Pi / Pi_dyn -> Action corrigee
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

try:
    import pandas as pd
    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier
except ModuleNotFoundError as exc:
    missing_name = exc.name or "une dependance requise"
    raise SystemExit(
        "Dependance manquante: "
        f"{missing_name}. Installez les packages avec "
        "`python3 -m pip install -r requirements.txt`."
    ) from exc


DEFAULT_FEATURES = ["sepal length (cm)", "sepal width (cm)"]
CLASS_MAPPING = {
    "setosa": 0,
    "versicolor": 1,
    "virginica": 2,
}


@dataclass
class ReflexConfig:
    uncertainty_margin: float = 0.08
    anomaly_zscore_limit: float = 2.75
    anomaly_force_zero: bool = True
    epsilon: float = 0.18


@dataclass
class RunResult:
    positive_class: str
    features: list[str]
    tau: float
    noise_fraction: float
    noise_std: float
    noisy_samples: int
    raw_accuracy: float
    corrected_accuracy: float
    raw_f1: float
    corrected_f1: float
    delta_raw: float
    delta_corrected: float
    dynamic_reflex_activated: bool
    confusion_raw: list[list[int]]
    confusion_corrected: list[list[int]]


def load_sepal_dataset(positive_class: str, features: Iterable[str]) -> tuple[pd.DataFrame, np.ndarray]:
    iris = load_iris(as_frame=True)
    frame = iris.frame.copy()
    selected = list(features)
    if not selected:
        raise ValueError("La liste de features ne peut pas etre vide.")

    y = (frame["target"] == CLASS_MAPPING[positive_class]).astype(int).to_numpy()
    X = frame[selected].copy()
    return X, y


def build_cognition() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )


def optimize_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    best_tau = 0.5
    best_score = -1.0
    for tau in np.linspace(0.1, 0.9, 161):
        preds = (probabilities >= tau).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_score:
            best_score = score
            best_tau = float(tau)
    return best_tau, best_score


def zscore_flags(train_frame: pd.DataFrame, test_frame: pd.DataFrame, limit: float) -> np.ndarray:
    means = train_frame.mean(axis=0)
    stds = train_frame.std(axis=0).replace(0, 1.0)
    zscores = ((test_frame - means) / stds).abs()
    return (zscores.max(axis=1) > limit).to_numpy()


def inject_perception_noise(
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    train_frame: pd.DataFrame,
    positive_class: str,
    noise_fraction: float,
    noise_std: float,
    random_state: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Injecte un bruit artificiel sur une partie du jeu de test.

    Cette fonction sert uniquement a la demonstration:
    - la cognition est entrainee sur un monde "propre"
    - au moment d'agir, une partie de la perception est degradee

    Le masque retourne represente un signal externe de sante capteur.
    Ce signal n'est pas utilise par le modele XGBoost lui-meme, mais il peut
    etre exploite par le systeme de reflexes pour corriger l'action.
    """
    if noise_fraction <= 0 or noise_std <= 0:
        return X_test.copy(), np.zeros(len(X_test), dtype=bool)

    rng = np.random.default_rng(random_state)
    X_noisy = X_test.copy()
    corrupted_mask = np.zeros(len(X_test), dtype=bool)

    n_corrupted = max(1, int(round(len(X_test) * noise_fraction)))

    # Pour rendre l'exemple visible, on degrade en priorite des exemples negatifs.
    # Dans le cas "setosa vs reste", cela augmente le risque de faux positifs:
    # la cognition seule devient plus fragile, tandis qu'un reflexe conservateur
    # peut couper certaines actions suspectes.
    candidate_idx = np.where(y_test == 0)[0]
    if len(candidate_idx) == 0:
        candidate_idx = np.arange(len(X_test))

    chosen = rng.choice(candidate_idx, size=min(n_corrupted, len(candidate_idx)), replace=False)
    corrupted_mask[chosen] = True

    train_stds = train_frame.std(axis=0).replace(0, 1.0)
    noise = rng.normal(loc=0.0, scale=noise_std, size=(len(chosen), X_test.shape[1]))

    # On oriente legerement la perturbation vers la classe positive.
    # Exemple "setosa": sepale plus court et un peu plus large.
    direction = np.zeros(X_test.shape[1], dtype=float)
    if positive_class == "setosa":
        if X_test.shape[1] >= 1:
            direction[0] = -0.9
        if X_test.shape[1] >= 2:
            direction[1] = 0.7

    scaled_noise = (noise + direction) * train_stds.to_numpy()
    X_noisy.iloc[chosen] = X_noisy.iloc[chosen].to_numpy() + scaled_noise
    return X_noisy, corrupted_mask


def raw_action(probabilities: np.ndarray, tau: float) -> np.ndarray:
    return (probabilities >= tau).astype(int)


def corrected_action(
    probabilities: np.ndarray,
    tau: float,
    anomaly_flags: np.ndarray,
    sensor_alert_flags: np.ndarray,
    config: ReflexConfig,
) -> np.ndarray:
    acts = raw_action(probabilities, tau)

    # Reflexe 1: zone d'incertitude autour du seuil -> comportement conservateur.
    uncertain = np.abs(probabilities - tau) <= config.uncertainty_margin
    acts[uncertain] = 0

    # Reflexe 2: echantillon atypique -> blocage de l'action positive.
    if config.anomaly_force_zero:
        acts[anomaly_flags] = 0

    # Reflexe 3: alerte capteur externe.
    # Idee importante: la cognition ne voit qu'un x degrade.
    # Les reflexes, eux, peuvent exploiter un autre canal d'information
    # (qualite capteur / integrite du signal) pour bloquer une action risquee.
    acts[sensor_alert_flags] = 0

    return acts


def action_delta(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true != y_pred))


def evaluate(
    y_true: np.ndarray,
    raw_pred: np.ndarray,
    corrected_pred: np.ndarray,
) -> tuple[float, float, float, float, list[list[int]], list[list[int]]]:
    raw_acc = float(accuracy_score(y_true, raw_pred))
    corrected_acc = float(accuracy_score(y_true, corrected_pred))
    raw_f1 = float(f1_score(y_true, raw_pred, zero_division=0))
    corrected_f1 = float(f1_score(y_true, corrected_pred, zero_division=0))
    raw_cm = confusion_matrix(y_true, raw_pred).tolist()
    corrected_cm = confusion_matrix(y_true, corrected_pred).tolist()
    return raw_acc, corrected_acc, raw_f1, corrected_f1, raw_cm, corrected_cm


def run_pipeline(
    positive_class: str = "setosa",
    features: Iterable[str] = DEFAULT_FEATURES,
    test_size: float = 0.25,
    epsilon: float = 0.18,
    noise_fraction: float = 0.0,
    noise_std: float = 0.0,
    random_state: int = 42,
) -> RunResult:
    X, y = load_sepal_dataset(positive_class=positive_class, features=features)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = build_cognition()
    model.fit(X_train, y_train)

    X_test_perceived, sensor_alert_flags = inject_perception_noise(
        X_test=X_test,
        y_test=y_test,
        train_frame=X_train,
        positive_class=positive_class,
        noise_fraction=noise_fraction,
        noise_std=noise_std,
        random_state=random_state,
    )

    train_prob = model.predict_proba(X_train)[:, 1]
    test_prob = model.predict_proba(X_test_perceived)[:, 1]

    tau, _ = optimize_threshold(y_train, train_prob)
    act = raw_action(test_prob, tau)
    delta = action_delta(y_test, act)

    reflexes = ReflexConfig(epsilon=epsilon)
    anomaly_flags = zscore_flags(X_train, X_test_perceived, reflexes.anomaly_zscore_limit)
    dynamic_reflex_activated = delta > reflexes.epsilon

    if dynamic_reflex_activated:
        act_star = corrected_action(test_prob, tau, anomaly_flags, sensor_alert_flags, reflexes)
    else:
        act_star = act.copy()

    delta_star = action_delta(y_test, act_star)
    raw_acc, corrected_acc, raw_f1, corrected_f1, raw_cm, corrected_cm = evaluate(
        y_test,
        act,
        act_star,
    )

    return RunResult(
        positive_class=positive_class,
        features=list(features),
        tau=tau,
        noise_fraction=noise_fraction,
        noise_std=noise_std,
        noisy_samples=int(sensor_alert_flags.sum()),
        raw_accuracy=raw_acc,
        corrected_accuracy=corrected_acc,
        raw_f1=raw_f1,
        corrected_f1=corrected_f1,
        delta_raw=delta,
        delta_corrected=delta_star,
        dynamic_reflex_activated=dynamic_reflex_activated,
        confusion_raw=raw_cm,
        confusion_corrected=corrected_cm,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Modele cognitif XGBoost avec seuil optimal et reflexes dynamiques."
    )
    parser.add_argument(
        "--positive-class",
        choices=sorted(CLASS_MAPPING),
        default="setosa",
        help="Classe positive y=1. Defaut: setosa.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=DEFAULT_FEATURES,
        help="Features a utiliser. Defaut: variables du sepale.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Proportion du jeu de test. Defaut: 0.25.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.18,
        help="Seuil d'activation des reflexes dynamiques.",
    )
    parser.add_argument(
        "--noise-fraction",
        type=float,
        default=0.0,
        help="Fraction du jeu de test degradee artificiellement pour la demonstration.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help="Intensite du bruit gaussien injecte sur les perceptions degradees.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Graine aleatoire pour split et bruit.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Affiche aussi une synthese lisible en plus du JSON.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Active une configuration de demonstration avec bruit perceptif et epsilon plus bas.",
    )
    return parser.parse_args()


def print_pretty_summary(result: RunResult, demo_enabled: bool) -> None:
    print("=== Cognitive AI / Sepal / XGBoost ===")
    print(f"Classe positive (R): {result.positive_class}")
    print(f"Perception Per(alpha): {', '.join(result.features)}")
    print(f"Seuil optimal tau: {result.tau:.3f}")
    print(f"Noise fraction: {result.noise_fraction:.3f}")
    print(f"Noise std: {result.noise_std:.3f}")
    print(f"Echantillons bruites: {result.noisy_samples}")
    print(f"Delta brut: {result.delta_raw:.4f}")
    print(f"Delta corrige: {result.delta_corrected:.4f}")
    print(f"Accuracy brute: {result.raw_accuracy:.4f}")
    print(f"Accuracy corrigee: {result.corrected_accuracy:.4f}")
    print(f"F1 brute: {result.raw_f1:.4f}")
    print(f"F1 corrigee: {result.corrected_f1:.4f}")
    print(f"Pi_dyn active: {result.dynamic_reflex_activated}")
    if not demo_enabled and result.noisy_samples == 0:
        print("Note: mode standard actif, donc pas de bruit injecte.")
        print("Pour voir une difference entre Act et Act*, lancez avec --demo.")
    elif demo_enabled:
        print("Note: mode demonstration actif, la perception a ete degradee artificiellement.")


def main() -> None:
    args = parse_args()
    noise_fraction = args.noise_fraction
    noise_std = args.noise_std
    epsilon = args.epsilon

    if args.demo:
        # Preset pedagogique: il rend la divergence cognition/reel plus visible.
        # Le but n'est pas d'ameliorer le modele, mais de montrer pourquoi des
        # reflexes externes peuvent reduire l'erreur d'action quand la perception
        # devient soudainement moins fiable.
        if noise_fraction == 0.0:
            noise_fraction = 0.35
        if noise_std == 0.0:
            noise_std = 1.25
        if epsilon == 0.18:
            epsilon = 0.04

    result = run_pipeline(
        positive_class=args.positive_class,
        features=args.features,
        test_size=args.test_size,
        epsilon=epsilon,
        noise_fraction=noise_fraction,
        noise_std=noise_std,
        random_state=args.random_state,
    )

    if args.pretty:
        print_pretty_summary(result, demo_enabled=args.demo)

    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
