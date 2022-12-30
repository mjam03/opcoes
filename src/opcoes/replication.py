import numpy as np

from opcoes.black_scholes import compute_greeks
from opcoes.black_scholes import compute_prices
from opcoes.simulation import generate_paths


def generate_dh_replications(pxs, deltas, op_pxs, rf, div, time_points):

    # we have 2 jobs:
    # - compute the value of the stock we hold at each time point (easy)
    # - compute the running cash balance from divs, delta rehedging (hard)

    # let's first compute the value of the delta we hold
    delta_values = pxs * deltas

    # now to compute the cash balance we need
    # - the initial balance from option premium and initial stock hedge
    # - cash flow from delta re-hedging (buying/selling stock)
    # - cash flow from divs on the delta we hold at each point
    # - grow our previous cash balance at the risk free rate

    # let's compute our initial cash outlay - init premium and delta cost
    start_cbs = op_pxs[:, 0] - deltas[:, 0] * pxs[:, 0]

    # let's compute the cash flows from re-hedging our delta
    delta_chgs = np.diff(deltas, axis=1)
    delta_rehedge_cfs = -delta_chgs * pxs[:, 1:]

    # let's compute dividend income
    div_cfs = deltas[:, :-1] * pxs[:, :-1] * div / time_points

    # now let's put it all together and run through each time point
    # init our cash balance and cashflow arrays
    cbs = [start_cbs]
    cfs = [np.zeros(len(pxs[:, 0]))]

    for drh, dcf in zip(delta_rehedge_cfs.T, div_cfs.T):

        # get previous cash balance
        prev_cash = cbs[-1]
        # future value it
        new_cash = prev_cash * (1 + rf / time_points)
        # add on new cash flows from delta rehedge and divs
        cf = drh + dcf
        cb = new_cash + cf
        # update cashflow and cash balance variables
        cfs.append(cf)
        cbs.append(cb)

    # transpose back
    cfs = np.array(cfs).T
    cbs = np.array(cbs).T

    # replication value is simply stock value plus cash balance
    repls = delta_values + cbs
    return repls


def generate_vh_replications(pxs, pf_vega, rf, div, time_points, hedge_greeks):

    # first we work out how much of the hedge instrument we need to hold
    # at each point to flatten the vega
    hedge_vega = hedge_greeks["vega"]
    ops_to_hold = -pf_vega / hedge_vega

    # using this we can figure out what our vega hedge portfolio delta is
    # along with the change in delta
    hedge_delta = hedge_greeks["delta"]
    hedge_deltas = ops_to_hold * hedge_delta
    hedge_delta_chgs = np.diff(hedge_deltas, axis=1)

    # let's compute the value of the delta we hold at each point
    # to hedge our vega hedge portfolio
    hedge_delta_values = -hedge_deltas * pxs

    # now we need to compute the value of the cashflows - they come from:
    # - initial prem of vega hedge and associated delta hedge
    # - rehedging the vega hedge's delta
    # - divs on the vega hedge's delta
    # - premiums from buy or selling options to rebalance the vega hedge

    # first let's work out delta hedge cash flows
    hedge_delta_rehedge_cfs = hedge_delta_chgs * pxs[:, 1:]

    # next let's compute div cashflows
    hedge_div_cfs = -hedge_deltas[:, :-1] * pxs[:, :-1] * div / time_points

    # next let's work out the re-hedge premiums
    # we do this by getting optiosn to trade each day * price on that day
    ops_to_trade = np.diff(ops_to_hold)
    hedge_pxs = hedge_greeks["px"]
    # we trade the quantity, so pay/receive the negative of the quantity
    rehedge_prems = -ops_to_trade * hedge_pxs[:, 1:]

    # init our starting cash balance and cash flows
    init_prems = -ops_to_hold[:, 0] * hedge_pxs[:, 0]
    init_stock_costs = ops_to_hold[:, 0] * hedge_delta[:, 0] * pxs[:, 0]
    init_cbs = init_prems + init_stock_costs

    hedge_cbs = [init_cbs]
    hedge_cfs = [np.zeros(len(pxs[:, 0]))]

    # now let's walk through each sim
    for h_drh, h_dcf, h_rhp in zip(
        hedge_delta_rehedge_cfs.T, hedge_div_cfs.T, rehedge_prems.T
    ):

        # get previous cash balance
        prev_cash = hedge_cbs[-1]
        # future value it
        new_cash = prev_cash * (1 + rf / time_points)
        # add on new cash flows which are:
        # - from div payments
        # - from re-hedging more options
        # - from delta-hedging the resulting structure
        cf = h_drh + h_dcf + h_rhp
        cb = new_cash + cf
        hedge_cfs.append(cf)
        hedge_cbs.append(cb)

    hedge_cfs = np.array(hedge_cfs).T
    hedge_cbs = np.array(hedge_cbs).T

    vh_repl = hedge_delta_values + hedge_cbs
    vh_pxs = ops_to_hold * hedge_pxs
    vh_vegas = ops_to_hold * hedge_vega
    return vh_pxs, vh_repl, vh_vegas


def generate_replications(pxs, deltas, call_pxs, risk_free, divs, points):

    # compute stock values at each point in time
    stock_values = pxs * deltas

    # compute cash flows from various activities
    # at each point we will add on cash flows to the future valued version
    # of last period's cash balance - this gives us our new cash balance

    # cash flow from delta hedging
    # compute cash generated from delta re-hedging
    delta_chgs = np.diff(deltas, axis=1)
    delta_rehedge_cfs = -delta_chgs * pxs[:, 1:]

    # compute dividend income
    div_cfs = deltas[:, :-1] * pxs[:, :-1] * divs / points

    # compute starting premium and delta cfs
    start_cbs = call_pxs[:, 0] - deltas[:, 0] * pxs[:, 0]

    # init our cash balance and cashflow arrays
    cbs = [start_cbs]
    cfs = [np.zeros(len(pxs[:, 0]))]

    for drh, dcf in zip(delta_rehedge_cfs.T, div_cfs.T):

        # get previous cash balance
        prev_cash = cbs[-1]
        # future value it
        new_cash = prev_cash * (1 + risk_free / points)
        # add on new cash flows
        cf = drh + dcf
        cb = new_cash + cf
        cfs.append(cf)
        cbs.append(cb)

    cfs = np.array(cfs).T
    cbs = np.array(cbs).T

    return stock_values, cbs, cfs


def run_delta_hedge_portfolio_sim(
    px,
    positions,
    rf,
    div,
    sigma,
    years,
    sims=10000,
    time_points=252,
    imp_vols=None,
    hedge_vols=None,
    greeks=["delta", "gamma", "vega"],
):

    # churn out sims using sigma for annual std dev of paths
    rets, pxs = generate_paths(px, rf - div, sigma, years, time_points, sims)

    # make sure we have implied vols
    # this allows us to specify different implied vols for pricing and hedging
    # compared to what we use for the brownian motion stock price simulation
    if imp_vols is None:
        imp_vols = sigma

    # now for each position in the positions list
    for p in positions:
        # compute the times vector used for valuing at each time point
        tenor = p["tenor"]
        start = tenor - years
        points = int(time_points * years) + 1
        times = np.linspace(start, tenor, points)[::-1]

        # compute prices and requested greeks for all times and stock paths
        K = p["strike"]
        call = p["cp"] == "C"
        results = compute_greeks(
            pxs,
            K,
            rf,
            div,
            imp_vols,
            times,
            call=call,
            hedge_vols=hedge_vols,
            greeks=greeks,
        )
        # add into posiitons dict
        p["results"] = results

    # now we have prices and greeks for each position
    # let's get overall portfolio numbers for the replication process
    all_greeks = ["px", "delta"] + greeks
    pf_greeks = {}
    for g in all_greeks:
        pf_g = sum([p["position"] * p["results"][g] for p in positions])
        pf_greeks[g] = pf_g

    # now for the replication we need at least pxs and deltas
    # if we wish to vega hedge then we also need vega

    # we can treat the delta hedging and vega hedging as separate processes
    # this is nicer as we will have distinct replication data sets for analysis
    pf_pxs = pf_greeks["px"]
    pf_deltas = pf_greeks["delta"]

    # now let's compute the delta replication
    repls = generate_dh_replications(
        pxs, pf_deltas, pf_pxs, rf, div, time_points
    )
    # return prices, returns, theo pf value, replication pf value and greeks
    return pxs, rets, pf_greeks["px"], repls, pf_greeks


def run_delta_hedge_sim(
    px,
    strike,
    rf,
    div,
    sigma,
    years,
    points,
    sims,
    iv=None,
    hedge_vol=None,
):

    # churn out sims using sigma for annual std dev of paths
    rets, pxs = generate_paths(px, rf - div, sigma, years, points, sims)
    # churn out option tvs, deltas and gammas based on iv with option
    # for different hedge vol
    if iv is None:
        iv = sigma

    call_pxs, deltas, gammas, vegas = compute_prices(
        pxs,
        strike,
        rf,
        div,
        iv,
        points,
        years,
        call=True,
        hedge_vol=hedge_vol,
    )

    # create replication values
    stock_values, cbs, cfs = generate_replications(
        pxs, deltas, call_pxs, rf, div, points
    )

    return pxs, rets, call_pxs, stock_values + cbs, deltas, gammas, vegas
