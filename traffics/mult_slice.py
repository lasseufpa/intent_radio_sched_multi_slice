import numpy as np

from sixg_radio_mgmt import Traffic


class MultSliceTraffic(Traffic):
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
        traffic_per_ue = np.zeros(self.max_number_ues)

        for slice in slice_req:
            if slice_req[slice] != {}:
                idx_ues = (slice_ue_assoc[int(slice[6]), :] == 1).nonzero()[0]
                traffic_per_ue[idx_ues] = (
                    self.rng.poisson(
                        slice_req[slice]["ues"]["traffic"], len(idx_ues)
                    )
                    * 1e6
                )  # Mbps

        return traffic_per_ue
