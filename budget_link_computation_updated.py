import os
import math
import csv
import pandas as pd
import rasterio
import numpy as np
from rasterio.merge import merge
from rasterio.plot import show
from sklearn.linear_model import LinearRegression

# Load all DEM files and merge them into a single dataset
dem_files = ['s01_e073_1arc_v3.tif', 's04_e055_1arc_v3.tif', 's05_e055_1arc_v3.tif', 's06_e055_1arc_v3.tif', 's06_e071_1arc_v3.tif', 's06_e072_1arc_v3.tif', 's07_e071_1arc_v3.tif', 's08_e056_1arc_v3.tif', 's08_e072_1arc_v3.tif', 's11_e056_1arc_v3.tif', 's20_e063_1arc_v3.tif', 's21_e055_1arc_v3.tif', 's21_e057_1arc_v3.tif', 's22_e055_1arc_v3.tif']
src_files_to_mosaic = []
for file in dem_files:
    src = rasterio.open(file)
    src_files_to_mosaic.append(src)
mosaic, out_trans = merge(src_files_to_mosaic)

# Get the elevation value at the given coordinate
lon, lat = 56.123, 11.456
row, col = rasterio.transform.rowcol(out_trans, lon, lat)
elevation = mosaic[0][row][col]
print(elevation)

def parse_coordinates(coord_string):
    direction = coord_string[-1]
    minutes = int(coord_string[-4:-2])
    degrees = int(coord_string[:-4])
    decimal_degrees = degrees + minutes / 60
    if direction in ('S', 'W'):
        decimal_degrees = -decimal_degrees
    return decimal_degrees

def load_waypoints(filename):
    waypoints = {}
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            waypoint = row[0]
            lat = parse_coordinates(row[1])
            lon = parse_coordinates(row[2])
            waypoints[waypoint] = (lat, lon)
    return waypoints

def load_receiver_sites(filename, dem_file):
    receiver_sites = {}
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            site = row[0].strip()
            lat = float(row[1].strip())
            lon = float(row[2].strip())
            antenna_height = int(row[3].strip())
            itu_minimum_received_power = int(row[4].strip())  # New field

            # Use DEM to estimate azimuth and tilt
            with rasterio.open(dem_file) as src:
                row, col = src.index(lon, lat)
                elevation = src.read(1, window=((row, row+1), (col, col+1)))
                elevation = elevation[0][0]
                x = np.array([[lat, lon, elevation]])
                model_azimuth = LinearRegression()
                model_tilt = LinearRegression()
                model_azimuth.fit(x, np.array([int(row[5].strip())]))
                model_tilt.fit(x, np.array([int(row[6].strip())]))
                azimuth = int(model_azimuth.predict(x)[0])
                tilt = int(model_tilt.predict(x)[0])
            receiver_sites[site] = {'coordinates': (lat, lon), 'antenna_height': antenna_height, 
                                     'itu_minimum_received_power': itu_minimum_received_power, 'azimuth': azimuth, 'tilt': tilt}
    return receiver_sites

def calculate_distance(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371e3
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi/2) * math.sin(delta_phi/2) + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2) * math.sin(delta_lambda/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R * c

def line_of_sight(distance, h1, h2):
    return distance <= 4.12 * (math.sqrt(h1) + math.sqrt(h2))

def free_space_path_loss(distance, frequency):
    return 32.4 + 20 * math.log10(distance) + 20 * math.log10(frequency)

def compute_coverage_quality(received_power, itu_minimum_received_power):
    if received_power >= itu_minimum_received_power:
        return "Strong"
    elif received_power >= itu_minimum_received_power - 10:
        return "Marginal"
    else:
        return "No Coverage"

def compute_link_quality(distance, received_power, itu_minimum_received_power):
    if distance > 20 and received_power >= itu_minimum_received_power + 6:
        return "Good"
    elif distance <= 20 and received_power >= itu_minimum_received_power - 5:
        return "Marginal"
    else:
        return "Poor"

def compute_link_budget(distance, frequency_mhz, transmitter_power_w, antenna_gain, receiver_gain, losses):
    wavelength = 3e8 / (frequency_mhz * 1e6)
    path_loss = free_space_path_loss(distance, frequency_mhz * 1e6)
    received_power_w = transmitter_power_w * antenna_gain * receiver_gain * (wavelength / (4 * math.pi * distance)) ** 2
    received_power_dbm = 10 * math.log10(1000 * received_power_w)
    link_budget_db = 10 * math.log10(transmitter_power_w * 1000) + antenna_gain + receiver_gain - path_loss - losses
    return received_power_dbm, link_budget_db, path_loss

def calculate_coverage_link_quality(waypoints, receiver_sites, frequency_mhz=1090, transmitter_power_w=125, antenna_gain=3.1, cable_loss=2, connector_loss=1, misc_losses=0):
    results = []
    for tx_name, tx_coord in waypoints.items():
        for rx_name, rx_data in receiver_sites.items():
            rx_coord = rx_data['coordinates']
            antenna_height = rx_data['antenna_height']
            itu_minimum_received_power = rx_data['itu_minimum_received_power']
            azimuth = rx_data['azimuth']
            tilt = rx_data['tilt']
            distance = calculate_distance(tx_coord, rx_coord)
            if line_of_sight(distance, antenna_height, antenna_height):
                received_power_dbm, link_budget_db, path_loss = compute_link_budget(distance / 1000, frequency_mhz, transmitter_power_w, antenna_gain, ITU_antenna_gain(azimuth, tilt), cable_loss + connector_loss + misc_losses)
                if received_power_dbm >= itu_minimum_received_power:
                    coverage_quality = compute_coverage_quality(received_power_dbm, itu_minimum_received_power)
                    link_quality = compute_link_quality(distance / 1000, received_power_dbm, itu_minimum_received_power)
                    results.append((tx_name + '-' + rx_name, round(distance / 1000, 2), round(received_power_dbm, 1), itu_minimum_received_power, coverage_quality, link_quality, round(link_budget_db, 1), round(path_loss, 1)))
    return pd.DataFrame(results, columns=['Transmitter-Receiver Pair', 'Distance (km)', 'Received Power Level (dBm)', 'ITU Min. Received Power (dBm)', 'Coverage Classification', 'Link Quality Classification', 'Link Budget (dB)', 'Path Loss (dB)'])

def create_coverage_link_quality_table(satisfactory_scenarios, receiver_sites, frequency_mhz=1090, transmitter_power_w=125, antenna_gain=3.1, cable_loss=2, connector_loss=1, misc_losses=0):
    if not satisfactory_scenarios.empty:
        df = pd.DataFrame(satisfactory_scenarios, columns=['Transmitter-Receiver Pair', 'Distance (km)', 'Received Power Level (dBm)', 'ITU Min. Received Power (dBm)', 'Coverage Classification', 'Link Quality Classification', 'Link Budget (dB)', 'Path Loss (dB)'])

        # Split the Transmitter-Receiver Pair column
        df[['Transmitter', 'Receiver']] = df['Transmitter-Receiver Pair'].str.split('-', expand=True)

        # Merge with receiver sites dataframe to get antenna height and ITU minimum received power
        receiver_sites_df = pd.DataFrame.from_dict(receiver_sites, orient='index').reset_index()
        receiver_sites_df = receiver_sites_df.rename(columns={'index': 'Receiver'})
        df = pd.merge(df, receiver_sites_df[['Receiver', 'antenna_height', 'itu_minimum_received_power', 'azimuth', 'tilt']], on='Receiver', how='left')

        # Compute link budget
        df['Link Budget (dB)'] = compute_link_budget(df['Distance (km)'], frequency_mhz, transmitter_power_w, antenna_gain, ITU_antenna_gain(df['azimuth'], df['tilt']), cable_loss + connector_loss + misc_losses)

        # Compute received power and link quality
        df['Received Power Level (dBm)'] = df.apply(lambda row: compute_link_budget(row['Distance (km)'], frequency_mhz, transmitter_power_w, antenna_gain, ITU_antenna_gain(row['azimuth'], row['tilt']), cable_loss + connector_loss + misc_losses)[0], axis=1)
        df['Link Quality Classification'] = df.apply(lambda row: compute_link_quality(row['Distance (km)'], row['Received Power Level (dBm)'], row['itu_minimum_received_power']), axis=1)

        # Compute coverage classification
        df['Coverage Classification'] = df.apply(lambda row: compute_coverage_quality(row['Received Power Level (dBm)'], row['itu_minimum_received_power']), axis=1)
        df['Coverage Classification'] = np.where(df['Coverage Classification'] == 'No Coverage', df['Coverage Classification'], df['Coverage Classification'] + ' (' + df['itu_minimum_received_power'].astype(str) + ' dBm)')

        # Select desired columns
        df = df[['Transmitter', 'Receiver', 'Distance (km)', 'Received Power Level (dBm)', 'Coverage Classification', 'Link Quality Classification', 'Link Budget (dB)']]
        return df
    else:
        return pd.DataFrame()

def main():
    waypoints = load_waypoints('waypoint_coordinates_updated.csv')
    receiver_sites = load_receiver_sites('receiver_sites.csv', 's11_e056_1arc_v3.tif')

    satisfactory_scenarios = calculate_coverage_link_quality(waypoints, receiver_sites, frequency_mhz=1090, transmitter_power_w=125, antenna_gain=3.1, cable_loss=2, connector_loss=1, misc_losses=0)

    df = create_coverage_link_quality_table(satisfactory_scenarios, receiver_sites)

    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
