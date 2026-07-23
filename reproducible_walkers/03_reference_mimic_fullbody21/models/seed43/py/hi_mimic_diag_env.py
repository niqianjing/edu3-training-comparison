import torch

from .hi_mimic_env import BaseEnv


class HIMimicDiagnosticEnv(BaseEnv):
    """Diagnostic-only environment that reports termination signals."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._termination_probe_step = 0
        print("XIAOHAI_TERMINATION_PROBE_BODY_IDS=", self.termination_contact_cfg.body_ids)
        print("XIAOHAI_TERMINATION_PROBE_BODY_NAMES=", self.termination_contact_cfg.body_names)
        print("XIAOHAI_CONTACT_SENSOR_SHAPE=", tuple(self.contact_sensor.data.net_forces_w_history.shape))

    def check_reset(self):
        if getattr(self, "_termination_probe_step", 0) < 10:
            net = self.contact_sensor.data.net_forces_w_history
            selected = net[:, :, self.termination_contact_cfg.body_ids]
            contact_peak = torch.max(torch.norm(selected, dim=-1), dim=1)[0]
            contact_any = torch.any(contact_peak > 1.0, dim=1)
            capture = self.local_capture_points_error_sum() if self.use_local_capture_points else self.global_capture_points_error_sum()
            print(
                "XIAOHAI_TERMINATION_SIGNAL",
                self._termination_probe_step,
                "contact_peak_min_mean_max=",
                float(contact_peak.min()), float(contact_peak.mean()), float(contact_peak.max()),
                "contact_envs=", int(contact_any.sum()),
                "capture_min_mean_max=",
                float(capture.min()), float(capture.mean()), float(capture.max()),
                "capture_envs=", int((capture > self.cfg.terminate.capture_points_distance_threshold).sum()),
                "threshold=", float(self.cfg.terminate.capture_points_distance_threshold),
            )
            self._termination_probe_step += 1
        return super().check_reset()
