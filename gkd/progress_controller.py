import math

class NativeKLProgress:
    """Update m only after four warm-up probe intervals have completed."""

    def __init__(self, interval=100, warmup_windows=4, beta=0.9, alpha=0.5, cutoff=0.05):
        if interval <= 0 or warmup_windows < 2 or not 0 <= beta < 1 or not 0 < alpha <= 1:
            raise ValueError("invalid progress-controller hyperparameters")
        self.interval = interval
        self.warmup_windows = warmup_windows
        self.beta = beta
        self.alpha = alpha
        self.cutoff = cutoff
        self.multiplier = 1.0
        self.ema = None
        self.speeds = []
        self.reference_speed = None
        self.probe_count = 0

    def observe(self, step, native_kl):
        if not math.isfinite(native_kl) or native_kl < 0:
            raise ValueError("native KL must be finite and nonnegative")
        old_ema = self.ema
        self.ema = native_kl if old_ema is None else self.beta * old_ema + (1 - self.beta) * native_kl
        speed = None
        if old_ema is not None:
            speed = max(0.0, (old_ema - self.ema) / max(old_ema, 1e-12))
            self.speeds.append(speed)
        if step > 0:
            self.probe_count += 1
        if self.probe_count == self.warmup_windows:
            # Use the first three speeds, independent of the fourth window.
            import statistics
            self.reference_speed = statistics.median(self.speeds[: self.warmup_windows - 1])
        elif self.probe_count > self.warmup_windows:
            ratio = min(1.0, speed / max(self.reference_speed, 1e-12))
            self.multiplier = min(
                self.multiplier, (1 - self.alpha) * self.multiplier + self.alpha * ratio
            )
            if self.multiplier < self.cutoff:
                self.multiplier = 0.0
        return {
            "step": step,
            "native_probe_kl": native_kl,
            "probe_ema_kl": self.ema,
            "relative_kl_speed": speed,
            "reference_speed": self.reference_speed,
            "progress_multiplier": self.multiplier,
            "probe_count": self.probe_count,
        }

    def state_dict(self):
        return {key: getattr(self, key) for key in (
            "multiplier", "ema", "speeds", "reference_speed", "probe_count"
        )}

    def load_state_dict(self, state):
        for key in self.state_dict():
            setattr(self, key, state[key])

