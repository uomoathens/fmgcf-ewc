from dataclasses import dataclass, field

import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))


@dataclass
class Client:
    name: str
    features: np.ndarray
    labels: np.ndarray
    group: np.ndarray


@dataclass
class TrainingSettings:
    rounds: int = 120
    local_steps: int = 5
    learning_rate: float = 0.2
    l2: float = 0.001
    miss_weight: float = 0.0
    positive_weight: float = 1.0
    clip: float = 1.0
    noise_multiplier: float = 0.0
    delta: float = 1e-5
    seed: int = 0


@dataclass
class TrainingResult:
    parameters: np.ndarray
    history: list = field(default_factory=list)
    epsilon: float = float("inf")
    rho: float = float("inf")


def predict(parameters, features):
    return _sigmoid(features @ parameters[:-1] + parameters[-1])


def miss_gap(probabilities, labels, group):
    positives0 = (labels == 1) & (group == 0)
    positives1 = (labels == 1) & (group == 1)
    if positives0.sum() == 0 or positives1.sum() == 0:
        return 0.0
    return float(np.mean(1 - probabilities[positives1]) - np.mean(1 - probabilities[positives0]))


def loss_and_gradient(parameters, features, labels, group, settings):
    weights = parameters[:-1]
    p = predict(parameters, features)
    sample_weight = np.where(labels == 1, settings.positive_weight, 1.0)
    normaliser = sample_weight.sum()
    eps = 1e-12
    task = -np.sum(sample_weight * (labels * np.log(p + eps) + (1 - labels) * np.log(1 - p + eps))) / normaliser
    residual = sample_weight * (p - labels) / normaliser
    grad_w = features.T @ residual + settings.l2 * weights
    grad_b = residual.sum()
    loss = task + 0.5 * settings.l2 * float(weights @ weights)
    if settings.miss_weight > 0:
        positives0 = (labels == 1) & (group == 0)
        positives1 = (labels == 1) & (group == 1)
        if positives0.sum() > 0 and positives1.sum() > 0:
            gap = miss_gap(p, labels, group)
            slope = p * (1 - p)
            d1 = -(features[positives1] * slope[positives1, None]).mean(axis=0)
            d0 = -(features[positives0] * slope[positives0, None]).mean(axis=0)
            b1 = -slope[positives1].mean()
            b0 = -slope[positives0].mean()
            grad_w = grad_w + settings.miss_weight * 2 * gap * (d1 - d0)
            grad_b = grad_b + settings.miss_weight * 2 * gap * (b1 - b0)
            loss += settings.miss_weight * gap ** 2
    return float(loss), np.concatenate([grad_w, [grad_b]])


def zcdp_epsilon(rounds, clients, noise_multiplier, delta):
    if noise_multiplier <= 0:
        return float("inf"), float("inf")
    rho = rounds / (2.0 * clients * noise_multiplier ** 2)
    epsilon = rho + 2.0 * np.sqrt(rho * np.log(1.0 / delta))
    return float(epsilon), float(rho)


def federated_train(clients, n_features, settings, monitor=None):
    rng = np.random.default_rng(settings.seed)
    parameters = np.zeros(n_features + 1)
    history = []
    for round_index in range(1, settings.rounds + 1):
        updates = []
        for client in clients:
            local = parameters.copy()
            for _ in range(settings.local_steps):
                _, gradient = loss_and_gradient(local, client.features, client.labels, client.group, settings)
                local -= settings.learning_rate * gradient
            update = local - parameters
            norm = np.linalg.norm(update)
            if settings.noise_multiplier > 0:
                update = update * min(1.0, settings.clip / (norm + 1e-12))
                update = update + rng.normal(0.0, settings.noise_multiplier * settings.clip, size=update.shape)
            updates.append(update)
        parameters = parameters + np.mean(updates, axis=0)
        losses = [loss_and_gradient(parameters, c.features, c.labels, c.group, settings)[0] for c in clients]
        entry = {"round": round_index, "mean_client_loss": float(np.mean(losses))}
        if monitor is not None:
            entry.update(monitor(parameters))
        history.append(entry)
    epsilon, rho = zcdp_epsilon(settings.rounds, len(clients), settings.noise_multiplier, settings.delta)
    return TrainingResult(parameters=parameters, history=history, epsilon=epsilon, rho=rho)


def centralised_train(features, labels, group, n_features, settings):
    parameters = np.zeros(n_features + 1)
    for _ in range(settings.rounds * settings.local_steps):
        _, gradient = loss_and_gradient(parameters, features, labels, group, settings)
        parameters -= settings.learning_rate * gradient
    return parameters


def platt(scores, labels, steps=2000, learning_rate=0.1):
    logit = np.log(np.clip(scores, 1e-6, 1 - 1e-6) / np.clip(1 - scores, 1e-6, 1 - 1e-6))
    a, b = 1.0, 0.0
    for _ in range(steps):
        p = _sigmoid(a * logit + b)
        a -= learning_rate * float(np.mean((p - labels) * logit))
        b -= learning_rate * float(np.mean(p - labels))
    return lambda values: _sigmoid(a * np.log(np.clip(values, 1e-6, 1 - 1e-6) / np.clip(1 - values, 1e-6, 1 - 1e-6)) + b)


def auc(scores, labels):
    labels = np.asarray(labels)
    positives = labels.sum()
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))
