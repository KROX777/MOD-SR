class GuidanceScheduler:
    """Schedules guidance steps during diffusion sampling."""

    def __init__(
        self,
        num_timesteps: int,
        t_min: float = 0.3,
        t_max: float = 0.7,
        max_guidance_steps: int = 5,
    ):
        """
        Args:
            num_timesteps: Total number of diffusion timesteps
            t_min: Minimum normalized timestep for guidance (late stage cutoff)
            t_max: Maximum normalized timestep for guidance (early stage cutoff)
            max_guidance_steps: Maximum number of guidance steps to apply
        """
        self.num_timesteps = num_timesteps
        self.t_min = t_min
        self.t_max = t_max
        self.max_guidance_steps = max_guidance_steps
        self._steps_done = 0

    def reset(self):
        """Reset step counter for new sampling run."""
        self._steps_done = 0

    def should_apply(self, t_idx: int) -> bool:
        """Check if guidance should be applied at current timestep."""
        if self._steps_done >= self.max_guidance_steps:
            return False

        normalized_t = t_idx / self.num_timesteps
        if not (self.t_min <= normalized_t <= self.t_max):
            return False

        self._steps_done += 1
        return True

    def get_stats(self) -> dict:
        """Get scheduler statistics."""
        return {
            "steps_done": self._steps_done,
            "max_steps": self.max_guidance_steps,
            "t_range": [self.t_min, self.t_max],
        }
