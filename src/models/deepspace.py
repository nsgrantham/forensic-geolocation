import numpy as np
import pandas as pd

from .points import SpatialPoints

class SpatialClassifier(object):

    def __init__(self, model, seeds):
        self.model = model
        self.seeds = seeds

    def fit(self, sp, to_categorical=False, **kwargs):
        """Fit self.model to sp.values collected from sp.coords/

        Parameters
        ---------
        sp : SpatialPoints
        to_categorical : bool (optional)
            Convert class vector to binary class matrix required by some models

        Returns
        -------
        None
        """
        y = self._assign_cells(sp).values
        if to_categorical:
            y = self._to_categorical(y)
        self.model.fit(sp.values.as_matrix(), y, **kwargs)

    def predict(self, values, **kwargs):
        """Predict most likely cell for each observation in sp.values.

        Parameters
        ----------
        sp : SpatialPoints

        Returns
        -------
        Array of class predictions (type depends on instance of model)
        """
        return self.model.predict(values.as_matrix(), **kwargs)

    def predict_proba(self, values, **kwargs):
        """Predict cell probabilities for each observation in sp.values.

        Parameters
        ----------
        sp : SpatialPoints

        Returns
        -------
        Array of class probabilities (type depends on instance of model)
        """
        return self.model.predict_proba(values.as_matrix(), **kwargs)

    def evaluate_likelihood(self, values, sp, **kwargs):
        """Evaluate likelihood over sp for each observation in values.

        Parameters
        ----------
        values : pandas DataFrame
        sp : SpatialPoints

        Returns
        -------
        pandas DataFrame
        """
        #idx = pd.MultiIndix.from_product([values.index, sp.ids], names=['ids', 'domain'])
        #likelihood = pd.DataFrame(index=idx, dtype=float)
        cells = self._assign_cells(sp)
        cells = pd.Series(cells, index=sp.ids, name='cell')
        proba = self.predict_proba(values, **kwargs)
        # Classifiers from keras and sklearn produce proba with different total
        # number of columns. In keras, the col number is always equalt to the
        # total number of seeds/cells, whereas in sklearn, the labels of empty
        # cells are not retained and the col number will be less. The 'classes_'
        # attribute lists the non-empty cells.
        if hasattr(self.model, 'classes_'):
            classes = self.model.classes_
        else:
            classes = self.seeds.ids
        proba = pd.DataFrame(data=proba.T, index=classes, columns=values.index)
        divide_by_cell_size = lambda x: x / x.count()
        likelihood = (cells.to_frame()
                       .join(proba, on='cell')
                       .fillna(0)
                       .groupby('cell')
                       .transform(divide_by_cell_size)
                       .unstack())
        likelihood.index.set_names([values.index.name, sp.ids.name], inplace=True)
        return likelihood

    def _assign_cells(self, sp):
        """Find nearest seed in self.seeds for each point in sp.

        Parameters
        ----------
        sp : SpatialPoints

        Returns
        -------
        pandas Index
        """
        dist = self.seeds.pairwise_distances_from(sp)
        cells = (dist.ix[dist.groupby(level=1).idxmin()]
                     .index.get_level_values(level=0))
        return cells

    def _to_categorical(self, y):
        """Convert class vector (with ints 0, 1, 2, ...) to binary class matrix.

        Utility from keras.utils.np_utils.

        Parameters
        ----------
        y : numpy ndarray

        Returns
        -------
        numpy ndarray
        """
        Y = np.zeros((len(y), self.seeds.shape[0]))
        for i in range(len(y)):
            Y[i, y[i]] = 1.
        return Y


class Geolocator(object):

    def __init__(self, domain):
        self.domain = domain

    def predict(self, likelihood):
        """Predict most likely location for each observation in values.

        Parameters
        ----------
        values : pandas DataFrame

        Returns
        -------
        SpatialPoints
        """
        likelihood = likelihood.sum(axis=1).rename('likelihood')
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

    def predict_regions(self, likelihood, p, **kwargs):
        """Prediction probability region(s) for each observation in values

        Parameters
        ----------
        values : pandas DataFrame
        p : float or array of float
            Quantile(s) at which to form probability region(s).

        Returns
        -------
        dict of dicts of SpatialPoints
            Outter-level keys are values.index and inner-level keys are
            the probabilities in p, each holding a SpatialPoints instance.
        """
        assert all(0. <= prob <= 1. for prob in p)
        likelihood = likelihood.sum(axis=1).rename('likelihood')
        probs = likelihood.groupby(level='ids').transform(lambda x: x / x.sum())
        cum_probs = (probs.groupby(level='ids', group_keys=False)
                .apply(lambda x: x.sort_values(ascending=False).cumsum()))
        sp_pred_regs = {id_: {} for id_ in probs.index.get_level_values('ids').unique()}
        for prob in p:
            points_within_prob = (cum_probs.groupby(level='ids', group_keys=False)
                    .apply(lambda x: x[(x < prob).shift(1).fillna(x[0] < prob)]))
            coords_within_prob = (points_within_prob.to_frame()
                    .drop('likelihood', axis=1)
                    .join(self.domain.coords, how='left'))
            str_prob = str(prob)
            for id_, coords in coords_within_prob.groupby(level='ids'):
                coords.index = coords.index.droplevel('ids')
                sp_pred_regs[id_][str_prob] = SpatialPoints(coords=coords)
        return sp_pred_regs

    def score(self, sp_preds, sp_true, miles=False, summarize=lambda x: x.mean()):
        """Compute distances between predicted and true locations.

        Parameters
        ----------
        sp_preds : dict of SpatialPoints
        sp_true : SpatialPoints

        Returns
        -------
        pandas Series
        """
        assert set(sp_preds) == set(sp_true.ids)
        errors = pd.Series(index=sp_true.ids, dtype=float, name='error')
        for id_, coords in sp_true.coords.groupby(level=0):
            sp_pred = sp_preds[id_]['most_likely']
            dist = sp_pred.pairwise_distances_from(SpatialPoints(coords=coords),
                    miles=miles)
            errors.set_value(id_, summarize(dist))
        return errors

    def score_areas(self, sp_pred, sp_true, areas='country', google_api_key=''):
        """Determine if predicted point lands in the correct area.

        Parameters
        ----------
        sp_pred : SpatialPoints
        sp_true : SpatialPoints
        area : string or array of strings

        Returns
        -------
        pandas Series or DataFrame
        """

        if isinstance(areas, str) or isinstance(areas, bytes):
            areas = [areas]

        if sp_true.areas.empty:
            sp_true._retrieve_areas_from_googlemaps(google_api_key)
        if sp_pred.areas.empty:
            sp_pred._retrieve_areas_from_googlemaps(google_api_key)

        # Accept a few convenient aliases for area names
        area_alias = {}
        if 'state' in areas:
            area_alias['administrative_area_level_1'] = 'state'
        if 'county' in areas:
            area_alias['administrative_area_level_2'] = 'county'
        if 'city' in areas:
            area_alias['locality'] = 'city'
        sp_true.areas.rename(columns=area_alias, inplace=True)
        sp_pred.areas.rename(columns=area_alias, inplace=True)

        shared_areas = set(sp_true.areas.columns).intersection(set(sp_pred.areas.columns))
        assert all(area in shared_areas for area in areas)

        match = pd.DataFrame(index=sp_true.ids, columns=areas, dtype=bool)
        for area in areas:
            match[area] = sp_true.areas[area] == sp_pred.areas[area]
        return match

    def score_regions(self, sp_pred_regs, sp_true):
        """Estimate coverage of predicted probability regions against true locations.

        Parameters
        ----------
        sp_pred_regs : dict of dicts of SpatialPoints
        sp_true : SpatialPoints

        Returns
        -------
        pandas DataFrame
        """
        assert set(sp_pred_regs) == set(sp_true.ids)
        p = list(next(iter(sp_pred_regs.values())))
        p.sort()
        coverage = pd.DataFrame(index=sp_true.ids, columns=p, dtype=bool)
        dist = sp_true.pairwise_distances_from(self.domain)
        nearest_domain_id = dict(dist.ix[dist.groupby(level=0).idxmin()].index)
        for sp_true_id, sp_pred_reg in sp_pred_regs.items():
            for prob, sp_pred in sp_pred_reg.items():
                in_reg = nearest_domain_id[sp_true_id] in sp_pred.ids
                coverage.set_value(sp_true_id, prob, in_reg)
        return coverage
