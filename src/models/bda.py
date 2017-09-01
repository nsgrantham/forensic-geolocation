import numpy as np
import pandas as pd

from .points import SpatialPoints

class GaussianKernelSmoother(object):

    def __init__(self, bandwidth):
        self.bandwidth = bandwidth

    def compute_weights(self, sp1, sp2):
        dist = sp1.pairwise_distances_from(sp2)
        kernel = np.exp(-0.5 * np.square(dist / self.bandwidth))
        weights = kernel.groupby(level=1).transform(lambda x: x / x.sum())
        return weights

    def generalized_cross_validate(self, sp):
        W = self.compute_weights(sp, sp).unstack(level=-1).as_matrix()
        X = sp.values.as_matrix()
        X_hat = np.dot(W, X)
        n = X.shape[0]
        z = np.sum(np.square(X_hat - X), axis=0) / (n * (1 - W.trace() / n)**2)
        gcv = pd.Series(data = z, index = sp.values.columns, name = self.bandwidth)
        return gcv

class BayesianDiscriminantAnalysis(object):

    def __init__(self, domain, smoothers):
        self.domain = domain
        self.smoothers = smoothers

    def group_taxa_by_optimal_bandwidths(self, sp):
        taxa = sp.values.columns
        gcv = pd.DataFrame(index = taxa)
        for smoother in self.smoothers:
            gcv = pd.concat([gcv, smoother.generalized_cross_validate(sp)], axis=1) 
        bandwidths = gcv.idxmin(axis=1)
        taxa_by_bandwidths = {}
        for bandwidth, series in bandwidths.groupby(bandwidths):
            taxa_by_bandwidths[bandwidth] = series.index.values
            print((bandwidth, len(series.index.values)))
        return taxa_by_bandwidths

    def estimate_occurrence_probabilities(self, sp, domain):
        taxa_by_bandwidths = self.group_taxa_by_optimal_bandwidths(sp)
        smoothers = {smoother.bandwidth: smoother for smoother in self.smoothers}
        probs_by_bandwidths = {}
        for bandwidth, taxa in taxa_by_bandwidths.items():
            X = sp.values[taxa].as_matrix()
            W = smoothers[bandwidth].compute_weights(domain, sp).unstack(level=-1).as_matrix()
            X_hat = np.dot(W, X)
            probs_by_bandwidths[bandwidth] = pd.DataFrame(data = X_hat, index = domain.ids, columns = taxa)
        probs = pd.concat(list(probs_by_bandwidths.values()), axis = 1)
        probs = probs[sp.values.columns]
        return probs
    
    def predict(self, likelihood):
        max_like = (likelihood.groupby(level='ids', group_keys=False)
                .apply(lambda x: x[x == x.max()]))
        max_like_coords = (max_like.to_frame()
                .drop('likelihood', axis=1)
                .join(self.domain.coords, how='left'))
        sp_preds = {id_: {} for id_ in
                max_like_coords.index.get_level_values('ids').unique()}
        for id_, coords in max_like_coords.groupby(level='ids'):
            coords.index = coords.index.droplevel('ids')
            sp_preds[id_]['most_likely'] = SpatialPoints(coords=coords)
        return sp_preds

    def score(self, sp_preds, sp_true, miles=False, summarize=lambda x: x.mean()):
        assert set(sp_preds) == set(sp_true.ids)
        errors = pd.Series(index=sp_true.ids, dtype=float, name='error')
        for id_, coords in sp_true.coords.groupby(level=0):
            sp_pred = sp_preds[id_]['most_likely']
            dist = sp_pred.pairwise_distances_from(SpatialPoints(coords=coords),
                    miles=miles)
            errors.set_value(id_, summarize(dist))
        return errors
    
    def _form_prediction_region(self, normalized_likelihood, threshold):
        assert 0. <= threshold <= 1.
        normalized_likelihood = normalized_likelihood.groupby(level='ids', group_keys=False).apply(lambda x: x.sort_values(ascending=False))
        cumsum_norm_like = normalized_likelihood.groupby(level='ids', group_keys=False).apply(lambda x: x.cumsum())
        in_region = (cumsum_norm_like.groupby(level='ids', group_keys=False)
                .apply(lambda x: (x < threshold).shift(1).fillna(x[0] < threshold)))
        coords_in_region = (normalized_likelihood[in_region]
                .to_frame()
                .drop('likelihood', axis=1)
                .join(self.domain.coords, how='left'))
        return coords_in_region

    def select_threshold(self, sp, probs, p, thresholds=np.arange(0.025, 1.0, step=0.025)):
        print("Selecting prediction region thresholds...")
        likelihood = self.evaluate_likelihood(sp.values, probs)
        normalized_likelihood = likelihood.groupby(level='ids').transform(lambda x: x / x.sum())
        sample_ids = normalized_likelihood.index.get_level_values('ids').unique()
        selected_thresholds = {}
        for prob in p:
            print(prob)
            str_prob = str(prob)
            covered = pd.DataFrame(index=sample_ids)
            for threshold in thresholds:
                print(threshold)
                coords_in_region = self._form_prediction_region(normalized_likelihood, threshold)
                sp_pred_regs = {id_: {} for id_ in sample_ids}
                for id_, coords in coords_in_region.groupby(level='ids'):
                    coords.index = coords.index.droplevel('ids')
                    sp_pred_regs[id_][str_prob] = SpatialPoints(coords=coords)
                covered = pd.concat([covered, self.score_regions(sp_pred_regs, sp)], axis=1)
            coverage = covered.mean(axis=0).values
            selected_thresholds[str_prob] = thresholds[np.argmax(coverage >= prob)]
        print("Done.")
        return selected_thresholds

    def predict_regions(self, likelihood, thresholds):
        normalized_likelihood = likelihood.groupby(level='ids').transform(lambda x: x / x.sum())
        sp_pred_regs = {id_: {} for id_ in normalized_likelihood.index.get_level_values('ids').unique()}
        for prob, threshold in thresholds.items():
            coords_in_region = self._form_prediction_region(normalized_likelihood, threshold)
            str_prob = str(prob)
            for id_, coords in coords_in_region.groupby(level='ids'):
                coords.index = coords.index.droplevel('ids')
                sp_pred_regs[id_][str_prob] = SpatialPoints(coords=coords)
        return sp_pred_regs

    def score_regions(self, sp_pred_regs, sp_true):
        assert set(sp_pred_regs) == set(sp_true.ids)
        p = list(next(iter(sp_pred_regs.values())))
        p.sort()
        coverage = pd.DataFrame(index=sp_true.ids, columns=p, dtype=bool)
        dist = sp_true.pairwise_distances_from(self.domain)
        nearest_domain_id = dict(dist.loc[dist.groupby(level=0).idxmin()].index)
        for sp_true_id, sp_pred_reg in sp_pred_regs.items():
            for prob, sp_pred in sp_pred_reg.items():
                in_reg = nearest_domain_id[sp_true_id] in sp_pred.ids
                coverage.set_value(sp_true_id, prob, in_reg)
        return coverage


    def evaluate_likelihood(self, values, probs):
        X = values.as_matrix()
        presence = X.T
        absence = 1 - X.T
        eps = 1e-10
        probs[probs < eps] = eps
        probs[probs > 1 - eps] = 1 - eps
        logp = np.log(probs.as_matrix())
        log1mp = np.log(1 - probs.as_matrix())
        L = np.dot(logp, presence) + np.dot(log1mp, absence)
        likelihood = pd.DataFrame(data=L.T - L.min(), index=values.index, columns=probs.index)
        likelihood = likelihood.stack().rename('likelihood')
        likelihood.index.names = ['ids', 'domain']
        return likelihood
