import folium
import geopandas as gpd
from pyspark.sql import functions as F, SparkSession
from IPython.display import display


def create_consumer_map(geojson, consumer_group_by_postcode, identifier, value, key):
    """
    Creates a Folium choropleth map shaded by a given metric across Australian postcodes.
    geojson: path or dict with postcode polygon boundaries.
    consumer_group_by_postcode: pandas DataFrame with one row per postcode.
    identifier: column name in consumer_group_by_postcode that matches the GeoJSON feature key.
    value: column name of the metric to shade (also used as the legend label).
    key: GeoJSON feature property name to join on (e.g. 'POA_CODE21').
    Displays the map inline; returns None.
    """
    # Centre on Australia; zoom level 4 fits the full continent
    m = folium.Map(location=[-25.2744, 133.7751], zoom_start=4)
    # Add Choropleth layer to the map
    folium.Choropleth(
        geo_data=geojson,
        name='choropleth',
        data=consumer_group_by_postcode,
        columns=[identifier, value],
        key_on='feature.properties.' + key,
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name= value
    ).add_to(m)
    display(m)