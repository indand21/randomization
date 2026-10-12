"""Randomisation algorithms and independently checkable allocation probabilities.

All permuted-block designs generate full balanced blocks and stop at the requested
recruitment count. Capping the final block is a separately named sensitivity.
The two-arm Beta-Bernoulli Thompson policy is sampled using its exact posterior
superiority probability, updated by beta-function recurrences.
"""
from __future__ import annotations
import hashlib
import math
from functools import lru_cache
import numpy as np
from numba import njit
from scipy.stats import binom, hypergeom

METHOD_IDS = {"simple": 0, "fixed_block": 1, "varying_block": 2,
              "minimization": 3, "stratified": 4, "thompson": 5,
              "bayesian": 6, "biased_coin": 7, "big_stick": 8,
              "capped_varying_block": 9}
BALANCE_METHODS = ["simple", "fixed_block", "varying_block", "minimization",
                   "stratified", "biased_coin", "big_stick"]
BLOCK_METHODS = {"fixed_block", "varying_block", "stratified",
                 "capped_varying_block"}
LABELS = {"simple": "Simple", "fixed_block": "Fixed block",
          "varying_block": "Variable block", "minimization": "Minimisation",
          "stratified": "Stratified block", "biased_coin": "Biased coin",
          "big_stick": "Big stick", "thompson": "Thompson sampling",
          "bayesian": "Tempered Bayesian", "capped_varying_block": "Capped variable block"}
PROFILE_NAMES = ["balanced", "skewed", "correlated"]

def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little")

def profile_parameters(profile, v):
    if profile == "balanced": return np.full(v, .5), 0.
    if profile == "skewed": return np.array([.15, .30, .45, .15, .30][:v]), 0.
    if profile == "correlated": return np.full(v, .5), .5
    if profile == "moderate_skew": return np.array([.25, .35, .45, .25, .35][:v]), 0.
    if profile == "high_correlation": return np.full(v, .5), .75
    if profile == "rare": return np.array([.05, .10, .20, .05, .10][:v]), 0.
    raise ValueError("Unknown covariate profile: " + profile)

def make_covariates(n, v, profile, iterations, seed):
    p, rho = profile_parameters(profile, v)
    rng = np.random.default_rng(seed)
    cov = (rng.random((iterations, n, v)) < p).astype(np.int8)
    if rho:
        # A patient-level mixture gives exactly rho pairwise correlation and
        # preserves Bernoulli(0.5) margins.
        copied = rng.random((iterations, n)) < rho
        common = (rng.random((iterations, n)) < .5).astype(np.int8)
        for j in range(v):
            cov[:, :, j] = np.where(copied, common, cov[:, :, j])
    return cov, p, rho

@njit(cache=False)
def beta_update(a, b, c, d, q, scale, arm_a, success):
    """Update q=P(Beta(a,b)>Beta(c,d)) and B(a+c,b+d)/(B(a,b)B(c,d))."""
    total = a+b+c+d
    if arm_a:
        if success:
            q += scale/a
            scale *= ((a+c)/total)*((a+b)/a)
            a += 1.
        else:
            q -= scale/b
            scale *= ((b+d)/total)*((a+b)/b)
            b += 1.
    else:
        if success:
            q -= scale/c
            scale *= ((a+c)/total)*((c+d)/c)
            c += 1.
        else:
            q += scale/d
            scale *= ((b+d)/total)*((c+d)/d)
            d += 1.
    q = min(1., max(0., q))
    return a,b,c,d,q,scale

@njit(cache=False)
def simulate_allocations(cov, population_p, method_id, block_size, seed,
                         true_pa=.6, true_pb=.4, keep_allocations=False):
    """Columns: nA, correct_guess, deterministic_fraction, max_std_cov_diff,
    successes, empty_strata, max_prefix_difference, arm_empty."""
    np.random.seed(seed)
    k, n, v = cov.shape
    result = np.zeros((k, 8), dtype=np.float64)
    allocation = np.zeros((k,n), dtype=np.int8) if keep_allocations else np.zeros((1,1), dtype=np.int8)
    n_strata = 2**v
    for r in range(k):
        a_count = 0
        counts = np.zeros((2,v,2), dtype=np.int64)
        stratum_counts = np.zeros(n_strata, dtype=np.int64)
        rem_a_s = np.zeros(n_strata, dtype=np.int64)
        rem_b_s = np.zeros(n_strata, dtype=np.int64)
        rem_a = 0
        rem_b = 0
        guess_sum = 0.
        deterministic = 0
        successes = 0
        prefix_max = 0
        a,b,c,d,q,scale = 1.,1.,1.,1.,.5,1./6.
        for t in range(n):
            sid = 0
            for j in range(v):
                sid += int(cov[r,t,j])*(2**j)
            stratum_counts[sid] += 1
            if method_id == 0:
                prob_a = .5
            elif method_id in (1,2,9):
                if rem_a+rem_b == 0:
                    bs = block_size
                    if method_id in (2,9):
                        bs = block_size + 2*(np.random.randint(0,3)-1)
                    if method_id == 9:
                        bs = min(bs, n-t)
                    rem_a = bs//2
                    rem_b = bs//2
                    if bs%2:
                        if np.random.random() < .5: rem_a += 1
                        else: rem_b += 1
                prob_a = rem_a/(rem_a+rem_b)
            elif method_id == 3:
                score_a = 0.
                score_b = 0.
                for j in range(v):
                    level = int(cov[r,t,j])
                    diff = counts[1,j,level]-counts[0,j,level]
                    score_a += abs(diff+1)
                    score_b += abs(diff-1)
                prob_a = .8 if score_a<score_b else (.2 if score_a>score_b else .5)
            elif method_id == 4:
                if rem_a_s[sid]+rem_b_s[sid] == 0:
                    rem_a_s[sid] = block_size//2
                    rem_b_s[sid] = block_size//2
                prob_a = rem_a_s[sid]/(rem_a_s[sid]+rem_b_s[sid])
            elif method_id == 5:
                prob_a = q
            elif method_id == 6:
                u = math.sqrt(q)
                w = math.sqrt(1.-q)
                prob_a = u/(u+w)
            elif method_id == 7:
                diff = 2*a_count-t
                prob_a = 2./3. if diff<0 else (1./3. if diff>0 else .5)
            else:  # big-stick design with maximum absolute imbalance 2
                diff = 2*a_count-t
                prob_a = 0. if diff>=2 else (1. if diff<=-2 else .5)
            guess_sum += max(prob_a,1.-prob_a)
            if prob_a < 1e-12 or prob_a > 1.-1e-12:
                deterministic += 1
            arm_a = np.random.random() < prob_a
            arm = 1 if arm_a else 0
            if arm_a: a_count += 1
            if keep_allocations: allocation[r,t] = arm
            for j in range(v):
                counts[arm,j,int(cov[r,t,j])] += 1
            if method_id in (1,2,9):
                if arm_a: rem_a -= 1
                else: rem_b -= 1
            if method_id == 4:
                if arm_a: rem_a_s[sid] -= 1
                else: rem_b_s[sid] -= 1
            prefix_max = max(prefix_max, abs(2*a_count-(t+1)))
            success = np.random.random() < (true_pa if arm_a else true_pb)
            successes += int(success)
            if method_id in (5,6):
                a,b,c,d,q,scale = beta_update(a,b,c,d,q,scale,arm_a,success)
        b_count = n-a_count
        cov_diff = np.nan
        if a_count and b_count:
            cov_diff = 0.
            for j in range(v):
                pa = counts[1,j,1]/a_count
                pb = counts[0,j,1]/b_count
                sd = math.sqrt(population_p[j]*(1.-population_p[j]))
                cov_diff = max(cov_diff, abs(pa-pb)/sd)
        empty = 0
        for s in range(n_strata):
            if stratum_counts[s] == 0: empty += 1
        result[r,0] = a_count
        result[r,1] = guess_sum/n
        result[r,2] = deterministic/n
        result[r,3] = cov_diff
        result[r,4] = successes
        result[r,5] = empty
        result[r,6] = prefix_max
        result[r,7] = int(a_count==0 or b_count==0)
    return result, allocation

def wilson_interval(events, repetitions, z=1.959963984540054):
    p = np.asarray(events,dtype=float)/repetitions
    den = 1.+z*z/repetitions
    centre = (p+z*z/(2.*repetitions))/den
    half = z*np.sqrt(p*(1.-p)/repetitions+z*z/(4.*repetitions**2))/den
    return np.maximum(0.,centre-half), np.minimum(1.,centre+half)

def fixed_distribution(n, block_size):
    r = n%block_size
    dist = {}
    for a in range(r+1):
        p = hypergeom.pmf(a,block_size,block_size//2,r)
        if p > 0: dist[2*a-r] = float(p)
    return dist

@lru_cache(maxsize=128)
def variable_distributions(max_n, block_size):
    sizes = [block_size-2,block_size,block_size+2]
    distributions = [{0:1.}]
    for r in range(1,max_n+1):
        out = {}
        for bs in sizes:
            if bs <= r:
                incoming = distributions[r-bs]
            else:
                incoming = {}
                for a in range(r+1):
                    prob = float(hypergeom.pmf(a,bs,bs//2,r))
                    if prob: incoming[2*a-r] = prob
            for diff, prob in incoming.items():
                out[diff] = out.get(diff,0.)+prob/len(sizes)
        distributions.append(out)
    return distributions

@lru_cache(maxsize=16)
def coin_distributions(max_n, method):
    out = [{0:1.}]
    for t in range(max_n):
        nxt = {}
        for diff,weight in out[-1].items():
            if method == "big_stick":
                pa = 0. if diff>=2 else (1. if diff<=-2 else .5)
            else:
                pa = 2./3. if diff<0 else (1./3. if diff>0 else .5)
            for new_diff,p in [(diff+1,pa),(diff-1,1.-pa)]:
                if p:
                    nxt[new_diff] = nxt.get(new_diff,0.)+weight*p
        out.append(nxt)
    return out

def exact_risk(method,n,block_size=4,threshold=.10):
    if method == "simple":
        a = np.arange(n+1)
        return float(binom.pmf(a,n,.5)[np.abs(2*a-n)/n>threshold].sum())
    if method == "fixed_block":
        dist = fixed_distribution(n,block_size)
    elif method == "varying_block":
        dist = variable_distributions(n,block_size)[n]
    elif method in ("biased_coin","big_stick"):
        dist = coin_distributions(n,method)[n]
    elif method == "capped_varying_block":
        return float((n%2)/n>threshold)
    else:
        return None
    return float(sum(p for diff,p in dist.items() if abs(diff)/n>threshold))

def design_features(method,n,v,profile,block_size):
    p,rho = profile_parameters(profile,v)
    b = block_size if method in BLOCK_METHODS else 0
    strata = 2**v if method == "stratified" else 1
    if method == "stratified":
        smallest = float(np.prod(np.minimum(p,1.-p)))
        if rho and v>1: smallest *= 1.-rho
    else:
        smallest = 1.
    return {"method":method,"N":n,"log_N":math.log(n),"N_inv":1./n,
            "n_factors":v,"block_size":b,
            "max_block_size":b+2 if method in ("varying_block","capped_varying_block") else b,
            "block_remainder":n%b if b else 0,
            "total_strata":strata,"log_strata":math.log(strata),
            "min_prevalence":float(p.min()),"max_prevalence":float(p.max()),
            "covariate_correlation":rho,"expected_min_stratum_size":n*smallest,
            "preferred_probability":.8 if method=="minimization" else (2./3. if method=="biased_coin" else .5),
            "mti":2 if method=="big_stick" else 0}
