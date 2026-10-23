"""Optional, seeded sensor imperfections for the LIO simulation mode.

Every effect is off by default so regression runs stay deterministic, and
every random draw comes from a generator seeded from ``seed`` so an
algorithm-evaluation run can be repeated exactly.  The models are deliberately
simple (white noise, bias random walk, hard clipping); report section 8 lists
that as a known simulation limit.
"""

from dataclasses import dataclass

import numpy as np

# Real-vehicle Point-LIO saturation (node_params.yaml): satu_acc 6 g,
# satu_gyro 35 rad/s.
IMU_ACC_SATURATION_G = 6.0
IMU_GYRO_SATURATION_RAD_S = 35.0


@dataclass(frozen=True)
class ImuNoiseConfig:
    enable_noise: bool = False
    enable_saturation: bool = False
    # Discrete-time white noise standard deviations at the 200 Hz output.
    gyro_noise_std: float = 7.0e-4   # rad/s
    accel_noise_std: float = 1.4e-3  # g
    # Bias random walk, per sqrt(second).
    gyro_bias_walk: float = 2.0e-5   # rad/s/sqrt(s)
    accel_bias_walk: float = 1.0e-5  # g/sqrt(s)
    seed: int = 20260723


class ImuNoiseModel:
    """Applies noise and saturation to one IMU sample at a time."""

    def __init__(self, config: ImuNoiseConfig):
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        self._gyro_bias = np.zeros(3)
        self._accel_bias = np.zeros(3)

    @property
    def active(self):
        return self.config.enable_noise or self.config.enable_saturation

    def apply(self, gyro, accel_g, dt):
        """Return (gyro rad/s, accel in g) after the configured effects."""
        gyro = np.asarray(gyro, dtype=np.float64)
        accel_g = np.asarray(accel_g, dtype=np.float64)
        cfg = self.config
        if cfg.enable_noise:
            root_dt = float(np.sqrt(max(dt, 0.0)))
            self._gyro_bias += self._rng.normal(0.0, cfg.gyro_bias_walk * root_dt, 3)
            self._accel_bias += self._rng.normal(0.0, cfg.accel_bias_walk * root_dt, 3)
            gyro = gyro + self._gyro_bias + self._rng.normal(0.0, cfg.gyro_noise_std, 3)
            accel_g = (
                accel_g + self._accel_bias + self._rng.normal(0.0, cfg.accel_noise_std, 3)
            )
        if cfg.enable_saturation:
            gyro = np.clip(gyro, -IMU_GYRO_SATURATION_RAD_S, IMU_GYRO_SATURATION_RAD_S)
            accel_g = np.clip(accel_g, -IMU_ACC_SATURATION_G, IMU_ACC_SATURATION_G)
        return gyro, accel_g


@dataclass(frozen=True)
class RangeNoiseConfig:
    enable: bool = False
    sigma_m: float = 0.02
    seed: int = 20260724


class RangeNoiseModel:
    """Gaussian error along each ray, applied to hit points in the sensor frame."""

    def __init__(self, config: RangeNoiseConfig):
        self.config = config
        self._rng = np.random.default_rng(config.seed)

    @property
    def active(self):
        return self.config.enable and self.config.sigma_m > 0.0

    def apply(self, points):
        """Scale each (n, 3) point along its ray from the sensor origin."""
        if not self.active or len(points) == 0:
            return points
        ranges = np.linalg.norm(points, axis=1)
        noisy = ranges + self._rng.normal(0.0, self.config.sigma_m, len(points))
        scale = np.where(ranges > 1e-9, noisy / np.maximum(ranges, 1e-9), 1.0)
        return (points * scale[:, np.newaxis]).astype(points.dtype, copy=False)
