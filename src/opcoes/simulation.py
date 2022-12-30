import numpy as np
from typing import Tuple


def generate_paths(
    spot: float,
    drift: float,
    sigma: float,
    years: float,
    points: int,
    sims: int,
) -> Tuple[np.ndarray, np.ndarray]:

    # create numpy PCG PRNG
    rng = np.random.default_rng(seed=123)
    # create random numbers
    dW = rng.normal(size=(sims, int(points * years)))
    # get our scaled random deviations
    scaled_dW = dW * (sigma / points**0.5)
    # add on the drift
    daily_devs = (drift / points) + scaled_dW
    # cumsum them as log returns additive
    cum_rets = np.cumsum(daily_devs, axis=1)
    # add in 0 at start for starting price
    cum_rets = np.insert(cum_rets, 0, 0, axis=1)
    # create price series
    pxs = spot * np.exp(cum_rets)
    # return the goodies
    return daily_devs, pxs
