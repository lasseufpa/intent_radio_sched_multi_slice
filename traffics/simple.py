import numpy as np

from sixg_radio_mgmt import Traffic


class SimpleTraffic(Traffic):
    def __init__(
        self,
        max_number_ues: int,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> None:
        super().__init__(max_number_ues, rng)

    def step(
        self,
        slice_ue_assoc: np.ndarray,
        slice_req: dict,
        step_number: int,
        episode_number: int,
    ) -> np.ndarray:
        return np.ones(self.max_number_ues) * 4
