import os
import folium
import pandas as pd
import seaborn as sns
from folium.plugins import HeatMap

# _______________________________________________________CONSTANTS______________________________________________________


OUTPUT_TEMPLATE = ("""
Statistical Analysis of Restaurants Data:
        
Chain restaurants:
    Mean Latitude: {}
    Mean Longitude: {}
            
    Latitude standard deviation: {}
    Longitude standard deviation: {}

Non-Chain restaurants:
    Mean Latitude: {}
    Mean Longitude: {}

    Latitude standard deviation: {}
    Longitude standard deviation: {}
""")

RESTAURANTS = ['restaurant', 'cafe', 'fast_food']
VANCOUVER_COORDINATES = [49.246292, -123.116226]


# _______________________________________________________GATHER DATA____________________________________________________

def check_data_exist():
    curr_dir = os.getcwd()

    # Check if data file exist
    input_dir = os.path.join(curr_dir, 'input')
    amenities_fp = os.path.join(input_dir, 'amenities-vancouver.json.gz')

    if not os.path.exists(amenities_fp):
        exit("Error: amenities-vancouver.json.gz is not present in /input directory.")

    # Create folder to store results
    output_fp = os.path.join(curr_dir, 'results')
    if not os.path.exists(output_fp):
        os.makedirs(output_fp)

    return amenities_fp, output_fp


# _____________________________________________________FILTER DATA______________________________________________________


def get_restaurants(locations):
    restaurants = locations[locations['amenity'].isin(RESTAURANTS)]
    return restaurants[restaurants['name'].notna()]


def get_restaurant_count(name, df):
    return df['total_branches'].loc[name]


def split_restaurants(restaurants):
    restaurants = restaurants.assign(total_branches=1)
    count = restaurants.groupby('name').count()
    restaurants['total_branches'] = restaurants['name'].apply(get_restaurant_count, df=count)

    chain_restaurants = restaurants[restaurants['total_branches'] > 1].reset_index()
    non_chain_restaurants = restaurants[restaurants['total_branches'] < 2].reset_index()
    return chain_restaurants, non_chain_restaurants


# __________________________________________________VISUALIZE DATA______________________________________________________

def heat_map(restaurants, van_map=None):
    if not van_map:
        van_map = folium.Map(location=VANCOUVER_COORDINATES, zoom_start=12)
    coordinates = restaurants[['lat', 'lon']].values.tolist()
    return HeatMap(coordinates).add_to(van_map)


def drop_pin(df, layer):
    layer.add_child(folium.Marker([df['lat'], df['lon']], popup=df['name']))


def pin_restaurants(restaurants, van_map=None):
    if not van_map:
        van_map = folium.Map(location=VANCOUVER_COORDINATES, zoom_start=12)
    marked = folium.map.FeatureGroup()
    restaurants.apply(drop_pin, axis=1, layer=marked)
    return van_map.add_child(marked)


def drop_mark(df, layer, color='black'):
    layer.add_child(folium.CircleMarker((df['lat'], df['lon']), radius=0.25, color=color))


def mark_restaurants(restaurants, color='Black', van_map=None):
    if not van_map:
        van_map = folium.Map(location=VANCOUVER_COORDINATES, zoom_start=12)
    incidents = folium.map.FeatureGroup()
    restaurants.apply(drop_mark, axis=1, layer=incidents, color=color)
    return van_map.add_child(incidents)


def restaurant_density(restaurants, color):
    sns.set()
    plot = sns.jointplot(x="lon", y="lat", data=restaurants, color=color)
    plot.set_axis_labels(xlabel='longitude', ylabel='latitude')
    return plot


# __________________________________________________IN-DEPT ANALYSIS____________________________________________________


def calculate_mean(restaurants):
    return restaurants['lat'].mean(), restaurants['lon'].mean()


def calculate_std(restaurants):
    return restaurants['lat'].std(), restaurants['lon'].std()


# _______________________________________________________MAIN___________________________________________________________


def main(amenities_file, out_directory, user_input):
    # ----------------------------------Step 1: Gather Data-----------------------------------
    data = pd.read_json(amenities_file, lines=True).dropna()

    # ----------------------------------Step 2: Filter Data-----------------------------------
    restaurants = get_restaurants(data)
    chain_restaurants, non_chain_restaurants = split_restaurants(restaurants)

    # ---------------------------------Step 3: Visualize Data---------------------------------

    # Chain and non-chain restaurants on map
    chain_mark = mark_restaurants(chain_restaurants, color='red')
    chain_non_chain_mark = mark_restaurants(non_chain_restaurants, color='blue', van_map=chain_mark)
    chain_non_chain_mark.save(os.path.join(out_directory, "chain-and-non-chain-locations.html"))

    # Chain restaurants heat map
    chain_heat_map = heat_map(chain_restaurants)
    chain_heat_map.save(os.path.join(out_directory, "chain-restaurants-heat-map.html"))

    # Non-chain restaurants heat map
    non_chain_heat_map = heat_map(non_chain_restaurants)
    non_chain_heat_map.save(os.path.join(out_directory, "non-chain-restaurants-heat-map.html"))

    # Chain restaurant density
    chain_desity = restaurant_density(chain_restaurants, color='red')
    chain_desity.savefig(os.path.join(out_directory, "chain-restaurants-density.png"))

    # Non-Chain restaurant density
    non_chain_desity = restaurant_density(non_chain_restaurants, color='blue')
    non_chain_desity.savefig(os.path.join(out_directory, "non-chain-restaurants-density.png"))

    # -------------------------------Step 4: In-dept Analysis---------------------------------

    # Mean coordinates
    chain_mean_lat, chain_mean_lon = calculate_mean(chain_restaurants)
    non_chain_mean_lat, non_chain_mean_lon = calculate_mean(non_chain_restaurants)

    # Standard deviation
    chain_lat_std, chain_lon_std = calculate_std(chain_restaurants)
    non_chain_lat_std, non_chain_lon_std = calculate_std(non_chain_restaurants)

    output_file = os.path.join(out_directory, "analysis.txt")
    with open(output_file, 'w') as f:
        output = OUTPUT_TEMPLATE.format(chain_mean_lat, chain_mean_lon, chain_lat_std, chain_lon_std,
                                        non_chain_mean_lat, non_chain_mean_lon, non_chain_lat_std, non_chain_lon_std)
        f.write(output)

    # ---------------------------------Step 5: Extra Feature----------------------------------

    # Extra feature to locate all Vancouver locations of the particular restaurant entered by the user.
    if user_input:
        locations = restaurants[restaurants['name'] == user_input]
        res_map = pin_restaurants(locations)
        res_map.save(os.path.join(out_directory, "{}-locations.html".format(user_input)))

        res_heat_map = heat_map(locations)
        res_heat_map.save(os.path.join(out_directory, "{}-heat-map.html".format(user_input)))

    return


if __name__ == '__main__':
    print("This program performs statistical analysis on all chain/non-chain restaurants in Vancouver.")
    print("In addition, it allows user to find all Vancouver locations of a particular restaurant.\n")

    amenities_path, output_path = check_data_exist()
    user_input = input("Optional - Enter restaurant name to find all its Vancouver locations (e.g. Starbucks): ")

    main(amenities_path, output_path, user_input)
