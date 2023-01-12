import folium
import folium.plugins
import armageddon


def plot_map(lat, lon, radii, map=None):
    """
    Plot blast radius on a map, as well as the population distribution
     (creating a new folium map instance if necessary).
    Parameters
    ----------

    lat: float
        latitude of circle to plot (degrees)
    lon: float
        longitude of circle to plot (degrees)
    radii: float
        list of radius (m)
    map: folium.Map
        existing map object

    Returns
    -------

    Folium map object

    Examples
    --------
     >>> map = plot_map(52.24377344901029, -2.0633566766367037, [32188.299005816258, 0.0, 0.0, 0.0])
     >>> map.save('../result.html')
    """
    locator = armageddon.PostcodeLocator()
    #  blast radius color
    color_map = ['blue', 'yellow', 'purple', 'red', 'black']
    color_index = 0
    for radi in radii:
        # Plot a circle on a map
        map = plot_circle(lat, lon, radi, map=map, color=color_map[color_index], fill_color=color_map[color_index])
        # calculate postcode within the radi
        distance = locator.norm(locator.postcodes[['Latitude', 'Longitude']].values, (lat, lon))
        data = locator.postcodes[distance < radi][['Postcode', 'Latitude', 'Longitude']]
        # get the population distribution
        data['Population'] = locator.get_population_of_postcode([data['Postcode']])[0]
        # plot the head map of population
        folium.plugins.HeatMap(data[['Latitude', 'Longitude', 'Population']].values.tolist(), radius=20,
                               min_opacity=0.2).add_to(map)
        color_index += 1
    # incidents = folium.map.FeatureGroup()
    # data['Population'] = data['Population'].apply(lambda x: x * (color_index + 1))
    # plot postcode on map
    # for index, row in data.iterrows():
    #     incidents.add_child(
    #         folium.CircleMarker(
    #             (row['Latitude'], row['Longitude']),
    #             radius=5,
    #             color=color_map[color_index],
    #             fill=True,
    #             fill_color=color_map[color_index],
    #             fill_opacity=0.6
    #         )
    #     )
    # map.add_child(incidents)
    return map


def plot_circle(lat, lon, radius, map=None, **kwargs):
    """
    Plot a circle on a map (creating a new folium map instance if necessary).

    Parameters
    ----------

    lat: float
        latitude of circle to plot (degrees)
    lon: float
        longitude of circle to plot (degrees)
    radius: float
        radius of circle to plot (m)
    map: folium.Map
        existing map object

    Returns
    -------

    Folium map object

    Examples
    --------

    >>> import folium
    >>> armageddon.plot_circle(52.79, -2.95, 1e3, map=None)
    """

    if not map:
        map = folium.Map(location=[lat, lon], control_scale=True, zoom_start=8)

    folium.Circle([lat, lon], radius, fill=True, fillOpacity=0.6, **kwargs).add_to(map)

    return map