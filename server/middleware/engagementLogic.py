import time
from constants import *

def handleEngagement(state):
    """
    Computes engagement using window-level metrics from NeuroAnalyzer.
    """

    z_scores = {}
    threshold_z = {}
    weighted_contrib = {}

    # Map logical proxies → analyzer metric keys
    proxy_map = {
        "pitch": "pitch_engagement",
        "yaw": "yaw_engagement",
        "visual": "visual_stability",
    }

    # --- 1. Compute z-scores for each metric ---
    for proxy, metric_key in proxy_map.items():
        stats = state.metrics.get(metric_key)
        if not stats:
            continue

        mean = stats["mean"]
        std = stats["std_dev"]

        if mean is None or std is None or std == 0:
            continue

        # Mean engagement score is the signal
        z = (mean - 1.0) / (std + 1e-6)
        z_scores[proxy] = z

        weighted_contrib[proxy] = z * state.weights.get(proxy, 0.0)

    if not z_scores:
        return  # Not enough data yet

    # --- 2. Compute threshold z-scores ---
    for proxy, metric_key in proxy_map.items():
        threshold_value = state.thresholds.get(proxy)
        stats = state.metrics.get(metric_key)

        if threshold_value is None or not stats:
            continue

        mean = stats["mean"]
        std = stats["std_dev"]

        if mean is None or std is None or std == 0:
            continue

        threshold_z[proxy] = (threshold_value - mean) / (std + 1e-6)

    # --- 3. Compute final engagement score ---
    engagement_score = sum(weighted_contrib.values())

    state.save_profile()

    # --- 4. Compute global engagement threshold ---
    global_threshold = 0.0
    for proxy, tz in threshold_z.items():
        global_threshold += tz * state.weights.get(proxy, 0.0)

    # --- 5. Compare engagement score to threshold ---
    if engagement_score < global_threshold:
        handle_false_alarm(state, engagement_score)

    # --- 6. Special case: yaw disengagement + keyboard + mouse = false ---
    if state.flags["phone_checking_mode"]:
        # User likely disengaged and inattentive
        # DISTRACTED_PING_SOUND_RETURN
        pass

    # --- 7. Update long-term variance ---
    update_long_term_variance(state, engagement_score)


def updateSessionStats(state, batch):
    """
    Updates state.session_stats using batch-level statistics.
    batch = {
        "gaze": {"mean": float, "n": int, "M2": float},
        "head": {...},
        "blink": {...},
        "expression": {...}
    }
    """

    for metric, stats in batch.items():
        batch_mean = stats["mean"]
        batch_n = stats["n"]
        batch_M2 = stats["M2"]

        if batch_n == 0:
            continue  # nothing to update

        roll = state.session_stats[metric]

        # If this is the first data for this metric
        if roll["n"] == 0:
            roll["mean"] = batch_mean
            roll["n"] = batch_n
            roll["M2"] = batch_M2
            roll["std"] = (batch_M2 / batch_n) ** 0.5 if batch_n > 1 else 0
            continue

        # Merge using Welford's algorithm
        old_n = roll["n"]
        new_n = old_n + batch_n

        delta = batch_mean - roll["mean"]

        # Update mean
        new_mean = roll["mean"] + (batch_n / new_n) * delta

        # Update M2
        new_M2 = (
            roll["M2"]
            + batch_M2
            + (delta ** 2) * (old_n * batch_n / new_n)
        )

        # Save back
        roll["mean"] = new_mean
        roll["n"] = new_n
        roll["M2"] = new_M2
        roll["std"] = (new_M2 / new_n) ** 0.5 if new_n > 1 else 0



def handle_false_alarm(state, current_values):
    """
    state: State object
    current_values: dict of latest proxy engagement scores,
                    e.g. {"gaze": 0.7, "blink": 0.3, ...}
    """

    z_scores = {}
    weighted_contrib = {}

    # Compute z-scores and weighted contributions
    for proxy, value in current_values.items():
        stats = state.session_stats[proxy]
        mean = stats["mean"]
        std = stats["std"]

        if mean is None or std is None or std == 0:
            continue

        z = (value - mean) / (std + 1e-6)
        z_scores[proxy] = z

        # Weighted contribution to engagement score
        weighted_contrib[proxy] = z * state.weights[proxy]

    if not weighted_contrib:
        return  # Not enough data

    # MAIN CHANGE:
    # The main offender is the metric with the MOST NEGATIVE weighted contribution
    main_offender = min(weighted_contrib, key=weighted_contrib.get)

    # Increment false alarm counter
    state.false_alarm_history[main_offender] += 1

    # If this proxy has caused enough false alarms, adjust it
    if state.false_alarm_history[main_offender] >= FALSE_ALARM_LIMIT:
        adjust_proxy(state, main_offender)
        state.false_alarm_history[main_offender] = 0
        state.save_profile()



def adjust_proxy(state, proxy):
    std = state.session_stats[proxy]["std"]

    if std is not None and std < LOW_VARIANCE_THRESHOLD:
        # Behavior is stable → threshold is too strict
        loosen_threshold(state, proxy)
    else:
        # Behavior is noisy → reduce weight
        adjust_weight(state, proxy)



def scale_threshold(threshold, loosen=True):
    if threshold > 0:
        # Positive threshold
        return threshold * (THRESHOLD_TIGHTEN if loosen else THRESHOLD_LOOSEN)
    elif threshold < 0:
        # Negative threshold
        return threshold * (THRESHOLD_LOOSEN if loosen else THRESHOLD_TIGHTEN)
    else:
        # Threshold is exactly zero — nudge it
        return 0.05 if loosen else -0.05

def loosen_threshold(state, proxy):
    key = THRESHOLD_KEYS[proxy]
    state.thresholds[key] = scale_threshold(state.thresholds[key], loosen=True)

def tighten_threshold(state, proxy):
    key = THRESHOLD_KEYS[proxy]
    state.thresholds[key] = scale_threshold(state.thresholds[key], loosen=False)
    

def adjust_weight(state, proxy):
    old_weight = state.weights[proxy]

    # Compute new weight for the offending metric
    new_weight = max(old_weight * WEIGHT_DECREMENT_FACTOR, MIN_WEIGHT)
    delta = old_weight - new_weight  # amount of weight we freed

    state.weights[proxy] = new_weight

    if delta <= 0:
        return  # nothing to redistribute

    # Collect other proxies to redistribute to
    other_proxies = [p for p in state.weights.keys() if p != proxy]

    # Sum of their current weights (for proportional redistribution)
    total_other_weight = sum(state.weights[p] for p in other_proxies)

    if total_other_weight == 0:
        # Edge case: everything else is zero, just leave it
        return

    # Redistribute delta proportionally to other metrics
    for p in other_proxies:
        share = state.weights[p] / total_other_weight
        state.weights[p] += delta * share



def update_long_term_variance(state, engagement_score):
    
    """
    Update long-term variance of each proxy engagement metric.

    This function runs every 10 minutes (i.e., if the last update was more than 10 minutes ago).
    It updates the long-term variance of each proxy engagement metric with the current fast variance.
    If the variance of a proxy engagement metric has dropped significantly and the engagement score is high, it tightens the threshold for that proxy engagement metric.

    :param state: State object containing current session stats and long-term variance
    :param engagement_score: Latest engagement score (sum of weighted proxy engagement scores)
    """
    now = time.time()

    # Only run every 10 minutes
    if now - state.last_variance_update_time < 600:
        return

    state.last_variance_update_time = now

    for proxy, stats in state.session_stats.items():
        fast_std = stats["std"]
        slow_std = state.long_term_variance[proxy]

        # First time: initialize slow variance
        if slow_std is None:
            state.long_term_variance[proxy] = fast_std
            continue

        # Compute improvement
        if fast_std is not None and slow_std is not None:
            improvement = slow_std - fast_std

            # If variance dropped significantly AND engagement is high
            if improvement > VARIANCE_IMPROVEMENT_THRESHOLD and engagement_score > HIGH_ENGAGEMENT_THRESHOLD:
                tighten_threshold(state, proxy)

        # Update slow variance for next cycle
        state.long_term_variance[proxy] = fast_std

    state.save_profile()


