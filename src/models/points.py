from warnings import warn

import os
import time

import googlemaps

import numpy as np
import pandas as pd

class SpatialPoints(object):

    def __init__(self, ids=None, coords=None, values=None, areas=None):
        ids = coords.index if coords is not None else \
                values.index if values is not None else \
                ids
        self.ids = ids
        
        if coords is None:
            coords = pd.DataFrame(index=ids, columns=['lat', 'lon'])
        self.coords = coords
        
        if values is None:
            values = pd.DataFrame(index=ids)
        self.values = values

        if areas is None:
            areas = pd.DataFrame(index=ids)
        self.areas = areas
        
        try:
            pd.concat([self.coords, self.values, self.areas], axis=1)
        except:
            raise ValueError('ids do not match between coords and values')

    def pairwise_distances_from(self, sp, miles=False):
        """Calculate pairwise distances between self.coords and sp.coords.
        
        Parameters
        ----------
        sp : SpatialPoints
        
        Returns
        -------
        pandas.Series
            MultiIndex Series with all pairwise distances between points
            in self.coords and those in s.
        """
        idx = pd.MultiIndex.from_product([self.coords.index, sp.coords.index])
        pairs = pd.concat([self.coords.add_suffix('1').reindex(idx, level=0),
                           sp.coords.add_suffix('2').reindex(idx, level=1)], axis=1)
        dist = self.great_circle(pairs['lat1'], pairs['lon1'], pairs['lat2'], pairs['lon2'], miles=miles)
        return dist

    def distances_from(self, sp, miles=False):
        """Calculate one-to-one distances between self.coords and sp.coords.
        
        Parameters
        ----------
        sp : SpatialPoints
            sp.ids must match self.ids
        
        Returns
        -------
        pandas.Series
            Series with distances between points in self and sp matched on ids
        """
        pairs = pd.concat([self.coords.add_suffix('1'), sp.coords.add_suffix('2')], axis=1)
        dist = self.great_circle(pairs['lat1'], pairs['lon1'], pairs['lat2'], pairs['lon2'], miles=miles)
        return dist
    
    def sample(self, n, reset_index=False):
        """Sample without replacement from ids.

        Parameters
        ----------
        n : int
        
        Returns
        -------
        SpatialPoints
        """
        ids = np.random.choice(self.ids, n, replace=False)
        coords = self.coords.ix[ids]
        values = self.values.ix[ids]
        if reset_index:
            coords.reset_index(drop=True, inplace=True)
            values.reset_index(drop=True, inplace=True)
        return SpatialPoints(coords=coords, values=values)

    def _retrieve_areas_from_googlemaps(self, google_api_key=''):
        """Query the Google Maps API to get information on the areas to which
        each point in self.coords belongs, such as country, ..., locality (city),
        and assign to self.areas.
        
        Parameters
        ----------
        google_api_key : string
            Must provide your personal API key. Get a key from here:
            https://developers.google.com/maps/documentation/geocoding/get-api-key
        """
        gmaps = googlemaps.Client(key=google_api_key)
        area_names = [
                'country', 'administrative_area_level_1',
                'administrative_area_level_2', 'administrative_area_level_3',
                'administrative_area_level_4','administrative_area_level_5',
                'locality', 'postal_code'
                ]
        self.areas = self.areas.reindex(columns=area_names)
        area_names = set(self.areas.columns)
        for id_, lat, lon in self.coords.itertuples():
            time.sleep(0.05)  # 20 queries per second (Google Maps API requires <50 qps)
            response = gmaps.reverse_geocode((lat, lon))
            if response:
                result = response[0]  # extract areas from top result
                for address_component in result['address_components']:
                    area_name = set(address_component['types']).intersection(area_names)
                    if area_name:
                        self.areas.set_value(id_, area_name, address_component['long_name'])
    
    @staticmethod
    def great_circle(lat1, lon1, lat2, lon2, miles=False):
        """Calculate great circle distance.
        
        http://www.johndcook.com/blog/python_longitude_latitude/
        
        Parameters
        ----------
        lat1, lon1, lat2, lon2: float or array of float
        miles : bool, optional
            Defines whether distance should be returned in miles, if True,
            or in kilometers.

        Returns
        -------
        float or array of float
            Distance(s) calculated between coordinates. For more info:
            https://en.wikipedia.org/wiki/Great-circle_distance
        """
        phi1 = np.deg2rad(90 - lat1)
        phi2 = np.deg2rad(90 - lat2)
        theta1 = np.deg2rad(lon1)
        theta2 = np.deg2rad(lon2)
        cos = (np.sin(phi1) * np.sin(phi2) * np.cos(theta1 - theta2) +
               np.cos(phi1) * np.cos(phi2))
        arc = np.arccos(np.clip(cos, -1, 1))
        arc_len = 3960 if miles else 6373  # kilometers
        return arc * arc_len

    @staticmethod
    def from_biom_table(table, coord_names=['lat', 'lon'], area_names=[], 
            ids_name='ids', verbose=False, dropna=True):
        """Construct SpatialPoints from BIOM.Table instance
        
        Parameters
        ----------
        tab : BIOM.Table
        
        Returns
        -------
        SpatialPoints
        """
        ids = table.ids(axis='sample')
        md = {id_: table.metadata(id_, axis='sample') for id_ in ids} 
        
        lat, lon = coord_names
        coords_dict = {
                id_: {
                    'lat': md[id_][lat],
                    'lon': md[id_][lon]
                    } 
                for id_ in md
                }
        coords = pd.DataFrame.from_dict(coords_dict, orient='index')
        coords.index.rename(ids_name, inplace=True)
        if dropna:
            n_total = coords.shape[0]
            coords.dropna(axis=0, how='any', inplace=True)
            n_dropped = n_total - coords.shape[0]
            if verbose:
                warn(
                    "Of {0} total points in BIOM.table, {1} points lack locational"
                    " metadata given by '{2}' and '{3}' and were dropped, leaving"
                    " {4} points for spatial analysis.".format(
                        n_total, n_dropped, lat, lon, n_total - n_dropped),
                    RuntimeWarning
                    )
        
        if area_names:
            areas_dict = {id_: {} for id_ in md}
            for id_ in md:
                for area in area_names:
                    areas_dict[id_][area] = md[id_][area]
            areas = pd.DataFrame.from_dict(areas_dict, orient='index')
            areas.index.rename(ids_name, inplace=True)
            areas = areas.loc[coords.index]
        else:
            areas = pd.DataFrame(index=coords.index)

        data = table.pa().matrix_data.todense().transpose()
        values = pd.DataFrame(data=data, index=table.ids(axis='sample'), 
                columns=table.ids(axis='observation'))
        values.index.rename(ids_name, inplace=True)
        values = values.loc[coords.index]
        
        return SpatialPoints(coords=coords, values=values, areas=areas)

    @property
    def shape(self):
        return self.values.shape
