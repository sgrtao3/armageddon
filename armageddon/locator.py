"""Module dealing with postcode information."""

import os
import sys
import math
import numpy as np
import pandas as pd
import re

def great_circle_distance(latlon1, latlon2):
    """
    Calculate the great circle distance (in metres) between pairs of
    points specified as latitude and longitude on a spherical Earth
    (with radius 6371 km).

    Parameters
    ----------

    latlon1: arraylike
        latitudes and longitudes of first point (as [n, 2] array for n points)
    latlon2: arraylike
        latitudes and longitudes of second point (as [m, 2] array for m points)

    Returns
    -------

    numpy.ndarray
        Distance in metres between each pair of points (as an n x m array)

    Examples
    --------

    >>> import numpy
    >>> fmt = lambda x: numpy.format_float_scientific(x, precision=3)}
    >>> with numpy.printoptions(formatter={'all', fmt}):
        print(great_circle_distance([[54.0, 0.0], [55, 0.0]], [55, 1.0]))
    [1.286e+05 6.378e+04]
    """

    R = 6371000
    # pre-handle the input data
    latlon1 = np.array(latlon1)
    latlon2 = np.array(latlon2)
    if latlon1.ndim == 1:
        latlon1 = latlon1.reshape(1, 2)
    if latlon2.ndim == 1:
        latlon2 = latlon2.reshape(1, 2)
    distance = np.empty((len(latlon1), len(latlon2)), float)

    index = 0
    # Haversine implementation
    # Calculate n points on one iteration, maybe this can be improved and eliminate the loop later.
    for lat2, lon2 in latlon2:
        lon1 = np.radians(latlon1[:, 1])
        lat1 = np.radians(latlon1[:, 0])
        lon2 = np.radians(lon2)
        lat2 = np.radians(lat2)
        lat = lat2 - lat1
        lng = lon2 - lon1
        d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
        distance[:, index] = 2 * np.arcsin(np.sqrt(d)) * R
        index += 1
    return distance


class PostcodeLocator(object):
    """Class to interact with a postcode database file."""

    def __init__(self, postcode_file='./resources/full_postcodes.csv',
                 census_file='./resources/population_by_postcode_sector.csv',
                 norm=great_circle_distance):
        """
        Parameters
        ----------

        postcode_file : str, optional
            Filename of a .csv file containing geographic
            location data for postcodes.

        census_file :  str, optional
            Filename of a .csv file containing census data by postcode sector.

        norm : function
            Python function defining the distance between points in latitude-longitude space.

        """
        # register distance function
        self.norm = norm
        # get and pre-handler postcode file
        BASE_PATH = os.path.dirname(__file__)
        self.postcodes = pd.read_csv(os.sep.join((BASE_PATH, postcode_file)))
        self.postcodes.dropna(axis=0, how='any', inplace=True)
        # drop any row with Nan value.
        self.postcodes['Postcode'].astype(str)
        # calculate and add sectors column through postcode.
        self.postcodes['Sectors'] = self.postcodes['Postcode'].apply(lambda x: self.get_sector_by_postcode(x))

        # get and pre-handler population file
        self.census = pd.read_csv(os.sep.join((BASE_PATH, census_file)))
        self.census.columns = ['date', 'geography', 'geography_code', 'total', 'rural_urban', 'population', 'males',
                               'females',
                               'communal', 'student', 'area', 'density']
        # Get rid of spaces in postcode strings
        self.census['geography'] = self.census['geography'].apply(
            lambda x: x.replace(' ', '') if x.count(' ') == 1 else x.replace('  ', ' '))
        # Get unit count of each sector
        unit_count = self.postcodes['Sectors'].value_counts()
        # we can assume uniform population distribution among the postcode units so
        # population =  sector population / total units in a sector
        self.census['unit_population'] = self.census.apply(
            lambda x: x['rural_urban'] / unit_count[x['geography']], axis=1
        )

    def get_sector_by_postcode(self, str):
        regex = r"^(((([A-Z][A-Z]{0,1})[0-9][A-Z0-9]{0,1}) {0,}[0-9])[A-Z]{2})$"
        matches = re.search(regex, str)
        if matches:
            return matches.group(2)
        else:
            return ''

    def get_postcodes_by_radius(self, X, radii, sector=False):
        """
        Return (unit or sector) postcodes within specific distances of
        input location.

        Parameters
        ----------
        X : arraylike
            Latitude-longitude pair of centre location
        radii : arraylike
            array of radial distances from X
        sector : bool, optional
            if true return postcode sectors, otherwise postcode units

        Returns
        -------
        list of lists
            Contains the lists of postcodes closer than the elements of radii to the location X.


        Examples
        --------

        >>> locator = PostcodeLocator()
        >>> locator.get_postcodes_by_radius((51.4981, -0.1773), [0.13e3])
        >>> locator.get_postcodes_by_radius((51.4981, -0.1773), [0.4e3, 0.2e3], True)
        """
        result = []
        distance = self.norm(self.postcodes[['Latitude', 'Longitude']].values, X)
        for radi in radii:
            if not sector:
                result.append(self.postcodes[distance < radi]['Postcode'].tolist())
            else:
                result.append(self.postcodes[distance < radi]['Sectors'].drop_duplicates().tolist())
        return result

    def get_population_of_postcode(self, postcodes, sector=False):
        """
        Return populations of a list of postcode units or sectors.

        Parameters
        ----------
        postcodes : list of lists
            list of postcode units or postcode sectors
        sector : bool, optional
            if true return populations for postcode sectors, otherwise postcode units

        Returns
        -------
        list of lists
            Contains the populations of input postcode units or sectors


        Examples
        --------

        >>> locator = PostcodeLocator()
        >>> locator.get_population_of_postcode([['SW7 2AZ','SW7 2BT','SW7 2BU', 'SW7 2DD']])
        >>> locator.get_population_of_postcode([['SW7  2']], True)
        """
        result = []
        # input will be  [['SW7 2AZ'], ['SW7 2AZ', 'SW7 2BU', 'SW7 2DD', 'SW7 5HG', 'SW7 5HQ']] if sector=False
        # we need to calculate the population for each entry in the list of lists and return the list of lists
        if not sector:
            for code_list in postcodes:
                code_list = [self.get_sector_by_postcode(i) for i in code_list]
                data = self.census[['geography', 'unit_population', 'rural_urban']]
                data = data[data['geography'].isin(code_list)]
                population_count = []
                for sector in code_list:
                    try:
                        # transfer postcode to sector
                        population_count.append(
                            data[data['geography'] == sector]['unit_population'].values[0])
                    except IndexError:
                        # print('no population data in {}'.format(name))
                        population_count.append(0)
                result.append(population_count)
        # things will like
        # [['SW7 1', 'SW7 2', 'SW7 3', 'SW7 4', 'SW7 5', 'SW7 9'],
        # ['SW7 1', 'SW7 2', 'SW7 3', 'SW7 4', 'SW7 5', 'SW7 9']]
        # if sector= True
        else:
            for list in postcodes:
                population_count = []
                for sector in list:
                    try:
                        population_count.append(
                            self.census[self.census['geography'] == sector]['rural_urban'].values[0])
                    except IndexError:
                        # print('no population data in {}'.format(sector))
                        population_count.append(0)
                result.append(population_count)
        return result


if __name__ == '__main__':
    locator = PostcodeLocator()
    postcode = locator.get_postcodes_by_radius((52.24377344901029, -2.0633566766367037),
                                               [32188.299005816258, 0.0, 0.0, 0.0],False)
    print(len(postcode[0]))
    import time
    time_start = time.time()
    population = locator.get_population_of_postcode(postcode)
    time_end = time.time()
    print(time_start - time_end)
