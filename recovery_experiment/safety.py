"""Substep measurements; contact whitelist and height semantics match env.py."""
import math
import mujoco
import numpy as np
from sim.env import SafeCubeEnv


class MeasuredEnv(SafeCubeEnv):
    def __init__(self, cfg, contact_tolerance=0.0):
        self.contact_tolerance = contact_tolerance
        self.monitoring = False
        self.samples = []
        super().__init__(cfg)

    def measure(self):
        depths = []
        for c in self.data.contact:
            a, b = int(c.geom1), int(c.geom2)
            other = b if a in self._red_geom_ids else a if b in self._red_geom_ids else None
            if other in self._arm_geom_ids or other in self._blue_geom_ids:
                depths.append(float(c.dist))
        grasped = self._blue_grasped()
        margin = (self.cfg.scene.ee_height_ceiling - float(self.data.xpos[self._blue_body_id, 2])) if grasped else None
        return dict(time=float(self.data.time), collision=any(d <= self.contact_tolerance for d in depths),
                    penetration=max([0.] + [-d for d in depths]), height=margin is not None and margin < 0,
                    height_margin=margin, ee_clearance_proxy=float(self.current_clearance()))

    def collision_clearance(self):
        """Minimum MuJoCo geom distance over exactly the prohibited pairs.

        For candidate selection, not a policy observation or runtime filter.
        Distances above 10 cm are censored to 10 cm.
        """
        best = .1
        pair = None
        segment = np.zeros(6)
        for red in self._red_geom_ids:
            for other in self._arm_geom_ids | self._blue_geom_ids:
                distance = float(mujoco.mj_geomDistance(self.model, self.data, red, other, .1, segment))
                if distance < best:
                    best, pair = distance, (int(red), int(other))
        return best, pair

    def begin_measurement(self):
        self.samples = []
        self.reset_safety = self.measure()
        self.monitoring = True

    def _update_grasp(self):
        super()._update_grasp()
        if self.monitoring:
            s = self.measure()
            self.samples.append(s)
            # Feed substep events into upstream termination, without incrementing
            # the success dwell counter at physics frequency.
            self._stats.red_contact |= s['collision']
            self._stats.ceiling_violation |= s['height']

    def summary(self):
        heights = [s['height_margin'] for s in self.samples if s['height_margin'] is not None]
        collision = any(s['collision'] for s in self.samples)
        height = any(s['height'] for s in self.samples)
        return dict(reset_collision=self.reset_safety['collision'], reset_height=self.reset_safety['height'],
                    collision=collision, height=height,
                    introduced_collision=collision and not self.reset_safety['collision'],
                    introduced_height=height and not self.reset_safety['height'],
                    max_penetration=max([0.] + [s['penetration'] for s in self.samples]),
                    min_height_margin=min(heights) if heights else None,
                    physics_samples=len(self.samples))
