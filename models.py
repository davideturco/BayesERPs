import pymc as pm
import numpy as np
from sklearn.linear_model import LinearRegression


class HierarchicalModel_complete_nc_type:
    """
    Non-centered (reparametrised) hierarchical model with varying intercepts and slopes, information is pooled amongst participant, items and ERP components
    Notice that compared to the model before, this one include a word type (contents vs function) contribution instead of a word contribution
    """

    def __init__(self, data):
        """
        sub = subject, t = type (i.e. word type), e = ERP component
        """
        self.data = data
        self.sub_idx = data["Participant"].values
        self.type_idx = data["tags"].values
        self.erp_idx = data["component"].values

        self.num_participants = len(self.data.Participant.unique())
        self.num_types = len(self.data.tags.unique())
        self.num_erp = len(self.data.component.unique())

        with pm.Model() as self.nc:
            self.chol_sub, _, _ = pm.LKJCholeskyCov(
                "chol_cov_sub", n=2, eta=2, sd_dist=pm.Exponential.dist(1.0), compute_corr=True
            )

            self.chol_t, _, _ = pm.LKJCholeskyCov(
                "chol_cov_t", n=2, eta=2, sd_dist=pm.Exponential.dist(1.0), compute_corr=True
            )

            self.chol_e, _, _ = pm.LKJCholeskyCov(
                "chol_cov_e", n=2, eta=2, sd_dist=pm.Exponential.dist(1.0), compute_corr=True
            )

            self.z_sub = pm.Normal("z_sub", 0, 1, shape=(2, self.num_participants))
            self.z_t = pm.Normal("z_t", 0, 1, shape=(2, self.num_types))
            self.z_e = pm.Normal("z_e", 0, 1, shape=(2, self.num_erp))

            self.ab_sub = pm.Deterministic("ab_sub", pm.math.dot(self.chol_sub, self.z_sub))
            self.ab_t = pm.Deterministic("ab_t", pm.math.dot(self.chol_t, self.z_t))
            self.ab_e = pm.Deterministic("ab_e", pm.math.dot(self.chol_e, self.z_e))

            self.a = pm.Normal("a", mu=0, sigma=1)
            self.b = pm.Normal("b", mu=0, sigma=0.5)

            self.surprisal = pm.Data("surprisal", self.data["surprisal"].values, mutable=False)

            self.mu_i = (self.a + self.ab_sub[0, self.sub_idx] + self.ab_t[0, self.type_idx] + self.ab_e[
                0, self.erp_idx]) + \
                        ((self.b + self.ab_sub[1, self.sub_idx] + self.ab_t[1, self.type_idx] + self.ab_e[
                            1, self.erp_idx]) *
                         self.surprisal)

            self.sigma_within = pm.HalfCauchy("sigma_within", 20)

            self.erp = pm.Normal("erp", mu=self.mu_i, sigma=self.sigma_within,
                                 observed=self.data["ERP"].values)

    def sample(self, num=3000, tune=1000, cores=4, target=.9, backend='default'):
        with self.nc:
            if backend == 'jax':
                import pymc.sampling_jax
                self.trace = pm.sampling_jax.sample_numpyro_nuts(num, tune=tune, chains=cores, target_accept=target)
            else:
                self.trace = pm.sample(num, tune=tune, cores=cores, return_inferencedata=True, target_accept=target)
        return self.trace


class LinearModel:
    def __init__(self, data):
        self.data = data

    def fit(self):
        coefs = []
        for i in self.data.Participant.unique():
            idx = self.data.Participant == i
            x = np.asarray(self.data.surprisal[idx]).reshape(-1, 1)
            y = self.data.ERP[idx]
            model = LinearRegression()
            reg = model.fit(x, y)
            coef = reg.coef_
            coefs.append(float(coef))
        return np.asarray(coefs)

