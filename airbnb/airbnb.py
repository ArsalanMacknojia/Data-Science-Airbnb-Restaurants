import os
import folium
import pandas as pd
from folium import plugins
import matplotlib.pyplot as plt
from folium.plugins import HeatMap
from sklearn.cluster import KMeans
from bounding_box import BoundingBox, get_bounding_box

# _______________________________________________________CONSTANTS______________________________________________________

MIN_AIRBNB_PRICE = 0
MAX_AIRBNB_PRICE = 100000
VANCOUVER_COORDINATES = [49.246292, -123.116226]
MAX_DISTANCE_FROM_AMENITIES = 5  # Airbnb locations within 5 kms radius of Amenities

BASIC_AMENITIES = ['restaurant', 'fast_food', 'fuel', 'atm', 'bank', 'pharmacy', 'clinic', 'cinema', 'bar']
CLUSTER_COLORS = ["red", "blue", "green", "purple", "orange", "darkred", "lightred", "cadetblue", "darkblue",
                  "darkgreen", "darkpurple", "pink", "lightgreen", "beige", "black"]


# _______________________________________________________GATHER DATA____________________________________________________


def check_data_exist():
    curr_dir = os.getcwd()

    # Check if data file exist
    input_dir = os.path.join(curr_dir, 'input')
    amenities_fp = os.path.join(input_dir, 'amenities-vancouver.json.gz')
    airbnb_fp = os.path.join(input_dir, 'airbnb-listings.csv.gz')

    if not os.path.exists(amenities_fp):
        exit("Error: amenities-vancouver.json.gz is not present in /input directory.")

    if not os.path.exists(airbnb_fp):
        exit("Error: airbnb-listings.csv.gz is not present in /input directory.")

    # Create folder to store results
    output_fp = os.path.join(curr_dir, 'results')
    if not os.path.exists(output_fp):
        os.makedirs(output_fp)

    return amenities_fp, airbnb_fp, output_fp


def get_user_input():
    # Get user input (Optional)
    min_price = input("Optional - Enter minimum Airbnb price: $")
    max_price = input("Optional - Enter maximum Airbnb price: $")

    input_val = dict()
    if min_price:
        input_val['mix_price'] = min_price
    if max_price:
        input_val['max_price'] = max_price

    return input_val


def get_listings(filename):
    columns = ['id', 'name', 'host_id', 'host_name', 'host_acceptance_rate', 'host_identity_verified', 'latitude',
               'neighbourhood_cleansed', 'city', 'country', 'zipcode', 'longitude', 'price', 'accommodates',
               'minimum_nights', 'number_of_reviews', 'review_scores_rating']
    # Read specified columns from CSV
    listings = pd.read_csv(filename, skipinitialspace=True, usecols=columns)

    # Rename columns
    listings.rename(columns={'neighbourhood_cleansed': 'neighbourhood'}, inplace=True)
    listings.rename(columns={'minimum_nights': 'min_nights'}, inplace=True)
    listings.rename(columns={'number_of_reviews': 'num_reviews'}, inplace=True)
    listings.rename(columns={'review_scores_rating': 'review_score'}, inplace=True)
    listings.rename(columns={'latitude': 'lat'}, inplace=True)
    listings.rename(columns={'longitude': 'lon'}, inplace=True)

    # Fix columns data types
    listings['price'] = listings['price'].str.replace('$', '').apply(pd.to_numeric, errors='coerce')
    listings['host_acceptance_rate'] = listings['host_acceptance_rate'].str.replace('%', '').apply(pd.to_numeric,
                                                                                                   errors='coerce')
    return listings


# _____________________________________________________FILTER DATA______________________________________________________


def filter_amenities(locations):
    amenities_types = dict()
    for amenity in BASIC_AMENITIES:
        amenities_types[amenity] = locations[locations['amenity'] == amenity]
    amenities_types['all'] = locations[locations['amenity'].isin(BASIC_AMENITIES)]

    return amenities_types


def filter_listings(listings, user_filter=None):
    print("\nTotal Airbnb listings: {}".format(len(listings.index)))

    # Choose places with minimum 10 reviews and 75+ average ratings.
    listings = listings[listings['review_score'].notna()]
    listings = listings[(listings['num_reviews'] > 10) & (listings['review_score'] > 75)]

    # Choose places with verified host only.
    listings = listings[listings['host_identity_verified'] == 't']

    # Filter place based on user price range. (If provided)
    if user_filter:
        min_price = user_filter.get('min_price', MIN_AIRBNB_PRICE)
        max_price = user_filter.get('max_price', MAX_AIRBNB_PRICE)
        listings = listings[(float(min_price) <= listings['price']) & (listings['price'] <= float(max_price))]

    print("Remaining Airbnb listings after filtering: {}".format(len(listings.index)))

    return listings


# __________________________________________________VISUALIZE DATA______________________________________________________


def plot_bar_chart(amenities_data, output):
    data = amenities_data['all'].groupby('amenity')['name'].count().reset_index(name='count')
    plt.bar(data['amenity'], data['count'])
    plt.xlabel('Amenities')
    plt.ylabel('Count')
    plt.savefig(output)


def heat_map(amenities_data, van_map=None):
    if not van_map:
        van_map = folium.Map(location=VANCOUVER_COORDINATES, zoom_start=12)
    coordinates = amenities_data[['lat', 'lon']].values.tolist()
    return HeatMap(coordinates).add_to(van_map)


def drop_pin(df, layer, color='black'):
    layer.add_child(folium.Marker([df['lat'], df['lon']], popup=df['name'], icon=folium.Icon(color=color)))


def pin_airbnb(airbnb_data, van_map=None):
    if not van_map:
        van_map = folium.Map(location=VANCOUVER_COORDINATES, zoom_start=12)
    pinned = plugins.MarkerCluster().add_to(van_map)
    airbnb_data.apply(drop_pin, axis=1, layer=pinned)
    return van_map.add_child(pinned)


def drop_mark(df, layer, color='black'):
    layer.add_child(folium.CircleMarker((df['lat'], df['lon']), radius=1, color=color))


def plot_amenities_cluster(amenities, van_map=None):
    if not van_map:
        van_map = folium.Map(location=VANCOUVER_COORDINATES, zoom_start=12)
    incidents = folium.map.FeatureGroup()
    for i in range(max(amenities['clusters'])):
        amenities[amenities['clusters'] == i].apply(drop_mark, axis=1, layer=incidents, color=CLUSTER_COLORS[i])
    return van_map.add_child(incidents)


# __________________________________________________IN-DEPT ANALYSIS____________________________________________________


def k_mean_cluster(amenities_data):
    model = KMeans(n_clusters=10)
    X = amenities_data[['lat', 'lon']]
    y = model.fit_predict(X)
    amenities = amenities_data.copy()
    amenities['clusters'] = y
    cluster_center = model.cluster_centers_

    return cluster_center, amenities


def get_ideal_listings(airbnb_locations, clusters_center, distance):
    ideal_locations = [get_bounding_box(center[0], center[1], distance) for center in clusters_center]
    ideal_listings = []
    for location in ideal_locations:
        listings = airbnb_locations[
            (airbnb_locations['lat'] > location.lat_min) & (airbnb_locations['lon'] > location.lon_min)]
        listings = listings[(listings['lat'] < location.lat_max) & (listings['lon'] < location.lon_max)]
        ideal_listings.append(listings)

    ideal_rentals = pd.concat(ideal_listings).drop_duplicates(keep=False)
    print("Total ideal Airbnb listings: {}".format(len(ideal_rentals.index)))

    return ideal_rentals


# _______________________________________________________MAIN___________________________________________________________


def main(amenities_file, airbnb_file, out_directory, user_filter):
    # ----------------------------------Step 1: Gather Data-----------------------------------
    amenities = pd.read_json(amenities_file, lines=True).dropna().reset_index(drop=True)
    airbnb = get_listings(airbnb_file)

    # ----------------------------------Step 2: Filter Data-----------------------------------
    amenities_locations = filter_amenities(amenities)
    airbnb_locations = filter_listings(airbnb, user_filter)

    # ---------------------------------Step 3: Visualize Data---------------------------------

    # Bar chart to display total number of each amenity
    plot_bar_chart(amenities_locations, os.path.join(out_directory, "amenities-bar-chart"))

    # Amenities heap map
    amenities_heat_map = heat_map(amenities_locations['all'])
    amenities_heat_map.save(os.path.join(out_directory, "amenities-heat-map.html"))

    # Amenities heat map + ALL Airbnb locations on map
    airbnb_map = pin_airbnb(airbnb_locations)
    amenities_heat_map_airbnb_pin = heat_map(amenities_locations['all'], airbnb_map)
    amenities_heat_map_airbnb_pin.save(os.path.join(out_directory, "amenities-heat-map-airbnb-pin.html"))

    # -------------------------------Step 4: In-dept Analysis---------------------------------

    # Amenities KMean cluster
    clusters_center, cluster_data = k_mean_cluster(amenities_locations['all'])
    cluster_map = plot_amenities_cluster(cluster_data)
    cluster_map.save(os.path.join(out_directory, "amenities-cluster.html"))

    # Amenities KMean clusters + IDEAL nearby Airbnb locations on map
    ideal_listings = get_ideal_listings(airbnb_locations, clusters_center, distance=MAX_DISTANCE_FROM_AMENITIES)

    amenities_cluster_ideal_listings = pin_airbnb(ideal_listings, cluster_map)
    amenities_cluster_ideal_listings.save(os.path.join(out_directory, "ideal-airbnb-listings.html"))

    return


if __name__ == '__main__':
    print("This program analyses Airbnb locations in Vancouver and recommend locations based on good amenities nearby.")
    print("In addition, it allows user to filter result based on price range and nearby locations of user's choice. \n")

    amenities_path, airbnb_path, output_path = check_data_exist()
    user_input = get_user_input()

    main(amenities_path, airbnb_path, output_path, user_input)
