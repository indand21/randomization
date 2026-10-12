"""Scientific checks against independent analytical calculations."""
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.stats import beta
from randomization_core import (METHOD_IDS, beta_update, exact_risk,
    make_covariates, simulate_allocations, wilson_interval)

class RandomisationChecks(unittest.TestCase):
    def test_fixed_incomplete_block_matches_hypergeometric(self):
        self.assertAlmostEqual(exact_risk("fixed_block",10,4,.1),1/3,places=13)
        self.assertEqual(exact_risk("fixed_block",20,4,.1),0.)

    def test_variable_blocks_do_not_force_terminal_balance(self):
        self.assertAlmostEqual(exact_risk("varying_block",10,4,.1),.19067215363511658,places=12)
        self.assertEqual(exact_risk("capped_varying_block",10,4,.1),0.)

    def test_monte_carlo_matches_exact_probabilities(self):
        cov,p,_=make_covariates(10,2,"balanced",30000,123)
        for method in ["simple","fixed_block","varying_block"]:
            raw,_=simulate_allocations(cov,p,METHOD_IDS[method],4,456)
            observed=np.mean(np.abs(2*raw[:,0]-10)/10>.1)
            exact=exact_risk(method,10,4,.1)
            self.assertLess(abs(observed-exact),6*np.sqrt(exact*(1-exact)/30000))

    def test_beta_recurrence_against_numerical_integration(self):
        rng=np.random.default_rng(55)
        a,b,c,d,q,s=1.,1.,1.,1.,.5,1/6
        for i in range(45):
            a,b,c,d,q,s=beta_update(a,b,c,d,q,s,bool(rng.integers(0,2)),bool(rng.integers(0,2)))
            if i%3==0:
                reference=quad(lambda x: beta.pdf(x,a,b)*beta.cdf(x,c,d),0.,1.,epsabs=1e-11)[0]
                self.assertAlmostEqual(q,reference,places=9)

    def test_allocation_conservation_and_big_stick_limit(self):
        for n in [9,10,31,60]:
            cov,p,_=make_covariates(n,3,"skewed",100,123)
            for method in METHOD_IDS:
                raw,alloc=simulate_allocations(cov,p,METHOD_IDS[method],4,456,.6,.4,True)
                np.testing.assert_array_equal(raw[:,0],alloc.sum(axis=1))
                self.assertTrue(np.all((raw[:,0]>=0)&(raw[:,0]<=n)))
                self.assertTrue(np.all((raw[:,1]>=.5)&(raw[:,1]<=1.)))
                if method=="big_stick": self.assertLessEqual(raw[:,6].max(),2.)

    def test_adaptive_null_is_symmetric(self):
        cov,p,_=make_covariates(60,2,"balanced",6000,123)
        for method in ["thompson","bayesian"]:
            raw,_=simulate_allocations(cov,p,METHOD_IDS[method],4,456,.5,.5)
            self.assertLess(abs(np.mean(raw[:,0])/60-.5),.025)

    def test_zero_event_interval_has_positive_width(self):
        lo,hi=wilson_interval(0,1000)
        self.assertAlmostEqual(float(lo),0.,places=12)
        self.assertGreater(float(hi),.003)
        self.assertLess(float(hi),.004)

    def test_covariate_generator_preserves_correlation_and_margins(self):
        cov,p,rho=make_covariates(500,3,"correlated",100,789)
        flat=cov.reshape(-1,3)
        self.assertLess(np.abs(flat.mean(axis=0)-p).max(),.015)
        self.assertLess(abs(np.corrcoef(flat[:,0],flat[:,1])[0,1]-rho),.025)

if __name__=="__main__": unittest.main(verbosity=2)
