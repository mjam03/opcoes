import numpy as np
import scipy.stats as stats


def bs_d_one(S, K, rf, div, sigma, delta_T):

    return (1 / (sigma * delta_T**0.5)) * (
        np.log(S / K) + (rf - div + (sigma**2) / 2) * delta_T
    )


def compute_prices(
    pxs, K, rf, div, sigma, points, years, call=True, hedge_vol=None
):

    # create time vector
    times = np.linspace(0, years, (points * years) + 1)[::-1]
    # calculate d1 and d2
    d_one = bs_d_one(pxs, K, rf, div, sigma, times)
    d_two = d_one - sigma * times**0.5
    # compute deltas and prices
    if call:
        deltas = stats.norm.cdf(d_one)
        prices = pxs * np.exp(-div * times) * deltas - stats.norm.cdf(
            d_two
        ) * K * np.exp(-rf * times)
        gammas = stats.norm.pdf(d_one) / (pxs * sigma * np.sqrt(times))
        vegas = pxs * stats.norm.pdf(d_one) * np.sqrt(times)

        if hedge_vol is not None:
            d_one = bs_d_one(pxs, K, rf, div, hedge_vol, times)
            deltas = stats.norm.cdf(d_one)
            gammas = stats.norm.pdf(d_one) / (pxs * hedge_vol * np.sqrt(times))
            vegas = pxs * stats.norm.pdf(d_one) * np.sqrt(times)
    else:
        deltas = stats.norm.cdf(d_one) - 1
        prices = stats.norm.cdf(-d_two) * K * np.exp(
            -rf * times
        ) + deltas * pxs * np.exp(-div * times)
        gammas = stats.norm.pdf(d_one) / (pxs * sigma * np.sqrt(times))
        vegas = pxs * stats.norm.pdf(d_one) * np.sqrt(times)
        if hedge_vol is not None:
            d_one = bs_d_one(pxs, K, rf, div, hedge_vol, times)
            deltas = stats.norm.cdf(d_one) - 1
            gammas = stats.norm.pdf(d_one) / (pxs * hedge_vol * np.sqrt(times))
            vegas = pxs * stats.norm.pdf(d_one) * np.sqrt(times)

    return prices, deltas, gammas, vegas


def compute_delta(d_one, div, times, call=True):

    # call delta given by:    exp(-qT) * N(d1)
    # put delta given by:     exp(-qT) * N(d1) - 1
    delta = np.exp(-div * times) * stats.norm.cdf(d_one)
    if call:
        return delta
    else:
        # put call parity innit
        return delta - 1


def compute_price(pxs, deltas, d_two, K, rf, div, times, call=True):

    # call: S * exp(-qT) * N(d1) - N(d2) * K * exp(-rT)
    # put:  K * exp(-rT) * N(-d2) - N(-d1) * S * exp(-qT)
    if call:
        return pxs * np.exp(-div * times) * deltas - stats.norm.cdf(
            d_two
        ) * K * np.exp(-rf * times)
    else:
        return K * np.exp(-rf * times) * stats.norm.cdf(
            -d_two
        ) + deltas * pxs * np.exp(-div * times)


def compute_gamma(pxs, d_one, vol, div, times):

    # gamma given by: exp(-qT) * n(d1) / (vol * S * sqrt(T))
    return (
        np.exp(-div * times)
        * stats.norm.pdf(d_one)
        / (vol * pxs * np.sqrt(times))
    )


def compute_vega(pxs, d_one, div, times):

    # vega given by: exp(-qT) * S * n(d1) * sqrt(T)
    return np.exp(-div * times) * pxs * stats.norm.pdf(d_one) * np.sqrt(times)


def compute_volga(pxs, K, rf, div, sigma, times):

    # we'll compute using first differences
    # compute vega if vol is 0.5vol higher
    d_one_up = bs_d_one(pxs, K, rf, div, sigma + 0.005, times)
    vega_up = compute_vega(pxs, d_one_up, div, times)
    # compute vega if vol is 0.5vol lower
    d_one_down = bs_d_one(pxs, K, rf, div, sigma - 0.005, times)
    vega_down = compute_vega(pxs, d_one_down, div, times)
    # return diff
    return vega_up - vega_down


def compute_greeks(
    pxs, K, rf, div, sigma, times, call=True, hedge_vols=None, greeks=["delta"]
):

    # dict holder for results
    results = {}
    # only valid greeks allowed for now
    valid_greeks = ["gamma", "vega", "volga"]
    # only compute greeks which are valid
    greeks = [g for g in greeks if g in valid_greeks]

    # calculate d1 and d2
    d_one = bs_d_one(pxs, K, rf, div, sigma, times)
    d_two = d_one - sigma * times**0.5

    # compute delta - need it anyway to compute prices
    deltas = compute_delta(d_one, div, times, call=call)
    # compute option prices
    prices = compute_price(pxs, deltas, d_two, K, rf, div, times, call=call)
    # add to results output
    results["px"] = prices
    results["delta"] = deltas

    # now compute extra greeks if requested
    # also recompute delta if requested with different vol
    if hedge_vols is not None or len(greeks) != 0:
        # if hedge vol different then need to recompute delta
        if hedge_vols is not None:
            # update vols to be what we requested
            vol = hedge_vols
            # recompute d1 with new vols
            d_one = bs_d_one(pxs, K, rf, div, vol, times)
            # recompute deltas
            deltas = compute_delta(d_one, div, times, call=call)
            # override in dict with new deltas
            results["delta"] = deltas
        else:
            # otherwise keep using prev vol for new greeks
            vol = sigma
        # now compute other greeks
        for g in greeks:
            if g == "gamma":
                # compute gamma and add to dict
                gammas = compute_gamma(pxs, d_one, vol, div, times)
                results["gamma"] = gammas
            if g == "vega":
                # compute vega and add to dict
                vegas = compute_vega(pxs, d_one, div, times)
                results["vega"] = vegas
            if g == "volga":
                # compute volga and add to dict
                volgas = compute_volga(pxs, K, rf, div, vol, times)
                results["volga"] = volgas

    # return results dictionary
    return results
