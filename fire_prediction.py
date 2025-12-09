import numpy as np
from sklearn.preprocessing import MinMaxScaler
import rasterio
from rasterio.transform import from_origin
from rasterio.plot import show
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import os
import encodings 
import time
from datetime import datetime
import re
from pyproj import Transformer
import glob

now = datetime.now()
year = now.year
month = now.month
day = now.day
hour = now.hour
minute = now.minute


# User-defined parameters for tuning out
grid_size =10                          # Example grid size
outlier_threshold = 2                   # Change this value to adjust outlier detection
downsampling_factor = 500               # Change this value to adjust downsampling
cwd = os.getcwd()
cwd_safe = re.sub(r'[<>"/\\|?* ]', '/', cwd)

# Create output directory for normalized TIFFs
output_directory = f"{cwd_safe}/GIS_Data/normalized_tiff"
print(f"{output_directory}")
os.makedirs(output_directory, exist_ok=True)
currentDay = f"{day}_{month}_{year}"




class DataSim:
    @staticmethod
    def generate_weather_data():
        weather_temp = np.random.uniform(30, 40, (grid_size, grid_size))
        weather_humidity = np.random.uniform(10, 30, (grid_size, grid_size))
        return weather_temp, weather_humidity

    @staticmethod
    def generate_fuel_loading():
        return np.random.uniform(0, 1, (grid_size, grid_size))




class ObtainActualData:

    @staticmethod
    def normalize_array(arr):
        """Normalize array to [0,1] safely."""
        arr = arr.astype("float32")
        arr[arr == -9999] = np.nan  # handle nodata
        min_val = np.nanmin(arr)
        max_val = np.nanmax(arr)

        if max_val == min_val:
            return np.zeros_like(arr, dtype="float32")

        return (arr - min_val) / (max_val - min_val)

    @staticmethod
    def load_raw_tiff(file_path):
        """Load raw TIFF and return array + metadata."""
        with rasterio.open(file_path) as src:
            data = src.read(1).astype("float32")
            transform = src.transform
            crs = src.crs
        return data, transform, crs

    @staticmethod
    def resample_to_grid(data, target_size):
        """Resample raster to (target_size x target_size)."""
        zoom_x = target_size / data.shape[0]
        zoom_y = target_size / data.shape[1]
        return zoom(data, (zoom_x, zoom_y), order=1)

    @staticmethod
    def save_normalized_tiff(output_path, data, crs, transform):
        """Save normalized raster as GeoTIFF."""
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype="float32",
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(data, 1)
        print(f"Saved: {output_path}")

    @staticmethod
    def process_and_normalize_raw_tiffs(raw_folder, output_folder, grid_size):
        """Normalize ALL TIFFs from raw_tiff → normalized_tiff."""
        os.makedirs(output_folder, exist_ok=True)
        files = [f for f in os.listdir(raw_folder) if f.lower().endswith(".tiff")]

        print(f"Found {len(files)} TIFF files in raw_tiff.")

        normalized_paths = {}

        for fname in files:
            raw_path = os.path.join(raw_folder, fname)
            norm_path = os.path.join(output_folder, f"normalized_{fname}")

            # Load
            raw_data, raw_transform, raw_crs = ObtainActualData.load_raw_tiff(raw_path)

            # Resample to grid_size
            resized = ObtainActualData.resample_to_grid(raw_data, grid_size)

            # Normalize
            normalized = ObtainActualData.normalize_array(resized)

            # Save output
            ObtainActualData.save_normalized_tiff(norm_path, normalized, raw_crs, raw_transform)

            normalized_paths[fname] = norm_path

        return normalized_paths

    @staticmethod
    def extract_raster_info(file_path):
        """
        Return raster array, min, and max values.
        This is used in FireSpreadModel to get basic raster stats.
        """
        data, _, _ = ObtainActualData.load_raw_tiff(file_path)
        #print(f"{data}, \n\n\n\n\n{_}, \n\n\n\n\n{_}") #                                                EPSG:4269 - Name: NAD83 (North American Datum 1983) | Type: Geographic coordinate system (GCS) | Units: Degrees (latitude and longitude) | Datum: NAD83 | Description: Used for mapping in North America. Coordinates are expressed as decimal degrees (latitude, longitude).
        min_val = np.nanmin(data)
        max_val = np.nanmax(data)
        return data, min_val, max_val

    @staticmethod
    def load_and_normalize_gis_data(file_path, grid_size, min_val=None, max_val=None):
        """
        Load a raster, resample to grid_size, and normalize.
        If min_val and max_val are provided, normalize using those.
        Otherwise, normalize to [0,1].
        """
        data, _, _ = ObtainActualData.load_raw_tiff(file_path)
        data_resized = ObtainActualData.resample_to_grid(data, grid_size)
        if min_val is not None and max_val is not None:
            normalized = (data_resized - min_val) / (max_val - min_val)
        else:
            normalized = ObtainActualData.normalize_array(data_resized)
        return normalized

    @staticmethod
    def remove_outliers(data, threshold=2):
        """
        Remove outliers by clipping to mean ± threshold * std.
        """
        mean = np.mean(data)
        std = np.std(data)
        lower = mean - threshold * std
        upper = mean + threshold * std
        return np.clip(data, lower, upper)




class FireSpreadModel:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size

    # ------------------- Feature Computation -------------------
    def compute_slope(self, elevation):
        slope = np.gradient(elevation, axis=(0, 1))
        slope_magnitude = np.sqrt(slope[0] ** 2 + slope[1] ** 2)
        return self.normalize_data(slope_magnitude)

    def combine_features(self, weather_temp, weather_humidity, slope, fuel_loading):
        """Compute static fire risk per cell (0-1) based on all inputs."""
        norm_temp = self.normalize_data(weather_temp)
        norm_humid = self.normalize_data(weather_humidity)
        norm_slope = self.normalize_data(slope)
        norm_fuel_load = self.normalize_data(fuel_loading)

        fire_risk = (norm_temp * 0.5 +
                     (1 - norm_humid) * 0.3 +
                     norm_slope * 0.5 +
                     norm_fuel_load * 0.3)

        # Normalize final risk to [0,1]
        fire_risk = (fire_risk - fire_risk.min()) / (fire_risk.max() - fire_risk.min())
        return fire_risk

    def normalize_data(self, data):
        scaler = MinMaxScaler()
        return scaler.fit_transform(data.flatten().reshape(-1, 1)).reshape(self.grid_size, self.grid_size)

    # ------------------- Fire Spread Simulation -------------------
    def spread_fire(self, fire_risk_map, start_location, decay=0.7, steps=5):
        """
        Probabilistic fire spread.
        - fire_risk_map: static 0-1 fire risk per cell
        - start_location: tuple (row, col)
        - decay: spread decay per step
        - steps: number of iterations
        Returns: continuous fire spread map (0-1)
        """
        fire_map = np.zeros_like(fire_risk_map, dtype=float)
        fire_map[start_location] = fire_risk_map[start_location]

        for step in range(steps):
            new_fire_map = fire_map.copy()
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if fire_map[i, j] > 0:
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                ni, nj = i + di, j + dj
                                if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                                    # Spread proportionally to neighbor risk and decay
                                    spread_value = fire_map[i, j] * fire_risk_map[ni, nj] * decay
                                    if spread_value > new_fire_map[ni, nj]:
                                        new_fire_map[ni, nj] = spread_value
            fire_map = new_fire_map

        # Normalize for visualization
        fire_map = fire_map / fire_map.max()
        return fire_map

    # ------------------- Combined Run -------------------
    def run_fire_spread(self, elevation_file=None, fuel_file=None,
                        temp_file=None, humidity_file=None,
                        start_location=None, outlier_threshold=2):
        """
        Run full simulation.
        Returns: (fire_risk_map, fire_spread_map)
        """
        # Load or simulate elevation
        if elevation_file:
            elevation = ObtainActualData.load_and_normalize_gis_data(elevation_file, self.grid_size)
        else:
            elevation = np.random.uniform(0, 100, (self.grid_size, self.grid_size))

        # Load or simulate fuel
        if fuel_file:
            fuel_loading = ObtainActualData.load_and_normalize_gis_data(fuel_file, self.grid_size)
        else:
            fuel_loading = DataSim.generate_fuel_loading()

        # Load or simulate weather
        if temp_file or humidity_file:
            if temp_file:
                weather_temp = ObtainActualData.load_and_normalize_gis_data(temp_file, self.grid_size)
            else:
                weather_temp, _ = DataSim.generate_weather_data()
            if humidity_file:
                weather_humidity = ObtainActualData.load_and_normalize_gis_data(humidity_file, self.grid_size)
            else:
                _, weather_humidity = DataSim.generate_weather_data()
        else:
            weather_temp, weather_humidity = DataSim.generate_weather_data()

        # Remove outliers
        elevation = ObtainActualData.remove_outliers(elevation, threshold=outlier_threshold)
        fuel_loading = ObtainActualData.remove_outliers(fuel_loading, threshold=outlier_threshold)
        weather_temp = ObtainActualData.remove_outliers(weather_temp, threshold=outlier_threshold)
        weather_humidity = ObtainActualData.remove_outliers(weather_humidity, threshold=outlier_threshold)

        # Compute static fire risk
        slope = self.compute_slope(elevation)
        fire_risk_map = self.combine_features(weather_temp, weather_humidity, slope, fuel_loading)

        # Fire start location
        if start_location is None:
            start_location = (self.grid_size // 2, self.grid_size // 2)

        # Run probabilistic spread
        fire_spread_map = self.spread_fire(fire_risk_map, start_location)

        #base_name = os.path.splitext(elevation_file)[0]
        base_name = os.path.splitext(os.path.basename(elevation_file))[0]#.replace("normalized_", "")
        # Replace illegal Windows characters and spaces with underscores
        base_name = re.sub(r'[<>:"/\\|?* ]', '_', base_name)


        #totalChangeLog = f"{cwd_safe}/GIS_Data/Change_Logs/{day}_{month}_prediction_processing_log"
        #directory_structure = f"{cwd_safe}/GIS_Data/{base_name}_Predictions/"
        
        
        #risk_prediction_output = f"{cwd_safe}/GIS_Data/Risk_Prediction_Output/PNG_Data_Type/{base_name}_Predictions"
        #spread_prediction_output = f"{cwd_safe}/GIS_Data/Spread_Prediction_Output/PNG_Data_Type/{base_name}_Predictions"
        risk_prediction_output = f"{cwd_safe}/GIS_Data/Risk_Prediction_Output/PDF_Data_Type/{base_name}_Predictions"
        spread_prediction_output = f"{cwd_safe}/GIS_Data/Spread_Prediction_Output/PDF_Data_Type/{base_name}_Predictions"


        #os.makedirs(f"{directory_structure}", exist_ok=True)
        #os.makedirs(f"{totalChangeLog}", exist_ok=True)
        os.makedirs(f"{risk_prediction_output}", exist_ok=True)
        os.makedirs(f"{spread_prediction_output}", exist_ok=True)

        # Optional visualization
        #DataVisualization.plot_2d(fire_risk_map, f"{base_name} Fire Risk Map", cmap="hot")
        #DataVisualization.plot_2d(fire_spread_map, f"{base_name} Fire Spread Map", cmap="hot")


        risk_fig = DataVisualization.plot_2d_return(fire_risk_map, f"{base_name} Fire Risk Map", cmap="hot")
        risk_fig.savefig(f"{risk_prediction_output}/Fire_Risk_Prediction_Map__{base_name}__{month}_{day}_{hour}_{minute}.png", format="png")        # save RISK to Fire_Risk_Prediction_Map_{base_name}.pdf
        risk_fig.savefig(f"{risk_prediction_output}/Fire_Risk_Prediction_Map__{base_name}__{month}_{day}_{hour}_{minute}.pdf", format="pdf")        # save RISK to Fire_Risk_Prediction_Map_{base_name}.pdf

        spread_fig = DataVisualization.plot_2d_return(fire_spread_map, f"{base_name} Fire Spread Map", cmap="hot")
        spread_fig.savefig(f"{spread_prediction_output}/Fire_Spread_Prediction_Map__{base_name}__{month}_{day}_{hour}_{minute}.png", format="png")      # save SPREAD to Fire_Spread_Prediction_Map_{base_name}.pdf
        spread_fig.savefig(f"{spread_prediction_output}/Fire_Spread_Prediction_Map__{base_name}__{month}_{day}_{hour}_{minute}.pdf", format="pdf")      # save SPREAD to Fire_Spread_Prediction_Map_{base_name}.pdf

        return fire_risk_map, fire_spread_map



class DataVisualization():
    def plot_2d_return(data, title, cmap):
        fig, ax = plt.subplots(figsize=(8, 8))   # create figure and axes
        cax = ax.imshow(data, cmap=cmap, origin='lower')
        fig.colorbar(cax, ax=ax, label='Elevation (ft)')
        ax.set_title(title)
        ax.set_xlabel('X (grid units)')
        ax.set_ylabel('Y (grid units)')
        
        return fig  # return the figure, don't call plt.show() here

    def plot_2d(data, title, cmap):
        #DataVisualization.plot_2d(data, title, cmap)
        plt.figure(figsize=(8, 8)) #used to be 10,7
        plt.imshow(data, cmap=cmap, origin='lower')
        plt.colorbar(label='Elevation (ft)')
        plt.title(title)
        plt.xlabel('X (grid units)')
        plt.ylabel('Y (grid units)')
        plt.show()

    def plot_3d(x, y, elevation, title, zlabel, cmap):
        #DataVisualization.plot_3d(x, y, elevation, title, zlabel, cmap)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        x, y = np.meshgrid(x, y)
        surf = ax.plot_surface(x, y, elevation, cmap=cmap, edgecolor='k')
        ax.set_xlabel('X (ft)')
        ax.set_ylabel('Y (ft)')
        ax.set_zlabel(zlabel)
        ax.set_title(title)
        plt.show()




class Functionality:
    @staticmethod
    def overlay_fire_predictions_on_tiff(
        tiff_path, 
        fire_risk_map, 
        fire_spread_map, 
        alpha=0.5, 
        cmap_risk='Reds', 
        cmap_spread='Oranges', 
        output_folder=None
    ):
        """
        Overlay fire risk and spread predictions over the original TIFF.

        Args:
            tiff_path (str): Path to original GeoTIFF.
            fire_risk_map (np.ndarray): 2D array of fire risk (0-1).
            fire_spread_map (np.ndarray): 2D array of fire spread (0-1).
            alpha (float): Transparency for overlay.
            cmap_risk (str): Colormap for risk overlay.
            cmap_spread (str): Colormap for spread overlay.
            output_folder (str, optional): Folder to save figure. If None, figure is not saved.

        Returns:
            matplotlib.figure.Figure: Figure object with overlay plot.
        """
        # Load original raster
        with rasterio.open(tiff_path) as src:
            raster_data = src.read(1)
            transform = src.transform
            crs = src.crs

        # Upsample predictions to match raster size
        risk_upsampled = zoom(fire_risk_map, (raster_data.shape[0] / fire_risk_map.shape[0],
                                              raster_data.shape[1] / fire_risk_map.shape[1]), order=1)
        spread_upsampled = zoom(fire_spread_map, (raster_data.shape[0] / fire_spread_map.shape[0],
                                                  raster_data.shape[1] / fire_spread_map.shape[1]), order=1)

        # Plot original raster
        fig, ax = plt.subplots(figsize=(10, 10))
        show(raster_data, ax=ax, cmap='terrain', origin='upper')

        # Overlay fire risk and spread
        risk_im = ax.imshow(risk_upsampled, cmap=cmap_risk, alpha=alpha, origin='upper')
        spread_im = ax.imshow(spread_upsampled, cmap=cmap_spread, alpha=alpha, origin='upper')

        # Colorbars
        cbar_risk = fig.colorbar(risk_im, ax=ax, fraction=0.046, pad=0.04)
        cbar_risk.set_label('Fire Risk')

        cbar_spread = fig.colorbar(spread_im, ax=ax, fraction=0.046, pad=0.08)
        cbar_spread.set_label('Fire Spread')

        ax.set_title(f"Overlay of Fire Risk & Spread on {os.path.basename(tiff_path)}")
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")

        # Save figure if folder provided
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(tiff_path))[0]
            output_file = os.path.join(output_folder, f"{base_name}_Overlay.pdf")
            fig.savefig(output_file, format='pdf')
            print(f"Overlay saved to {output_file}")

        return fig

    @staticmethod
    def plot_latlon_on_tiff(tiff_path, lat, lon, marker_color='red', marker_size=50, title=None):
        """
        Plot a latitude/longitude point on a GeoTIFF raster.

        Args:
            tiff_path (str): Path to the GeoTIFF file.
            lat (float): Latitude of the point.
            lon (float): Longitude of the point.
            marker_color (str): Color of the marker.
            marker_size (int): Size of the marker.
            title (str): Optional title for the plot.

        Returns:
            matplotlib.figure.Figure: Figure object with the plot.
        """
        with rasterio.open(tiff_path) as src:
            raster_data = src.read(1)
            transform = src.transform
            crs = src.crs

        # Transform lat/lon to raster CRS if needed
        if crs.is_geographic:
            x, y = lon, lat
        else:
            transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            x, y = transformer.transform(lon, lat)

        # Convert map coordinates to row/col in raster
        col, row = ~transform * (x, y)
        row, col = int(round(row)), int(round(col))

        # Plot raster
        fig, ax = plt.subplots(figsize=(8, 8))
        show(raster_data, ax=ax, cmap='terrain', origin='upper')

        # Save figure for debugging/testing
        os.makedirs(f"{cwd_safe}/1testingmyshit/", exist_ok=True)
        fig.savefig(f"{cwd_safe}/1testingmyshit/conifer_{lat}_{lon}.pdf", format="pdf")

        # Plot the marker
        ax.scatter(col, row, c=marker_color, s=marker_size, edgecolor='black', label=f"Lat: {lat}, Lon: {lon}")
        if title:
            ax.set_title(title)
        ax.legend()

        return fig

    @staticmethod
    def get_latlon_bounds(tiff_path):
        """
        Returns the geographic bounding box (lat/lon) of a GeoTIFF.

        Args:
            tiff_path (str): Path to the GeoTIFF file.

        Returns:
            dict: {'min_lat': ..., 'max_lat': ..., 'min_lon': ..., 'max_lon': ...}
        """
        with rasterio.open(tiff_path) as src:
            width, height = src.width, src.height
            transform = src.transform
            crs = src.crs

            # Raster corner coordinates in raster CRS
            corners = [
                (0, 0),           # top-left
                (width, 0),       # top-right
                (0, height),      # bottom-left
                (width, height)   # bottom-right
            ]

            # Convert corners to map coordinates
            map_coords = [transform * (col, row) for col, row in corners]

            # Transform to lat/lon if needed
            if crs.is_geographic:
                lons, lats = zip(*map_coords)
            else:
                transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                lons, lats = zip(*[transformer.transform(x, y) for x, y in map_coords])

            bounds = {
                'min_lat': min(lats),
                'max_lat': max(lats),
                'min_lon': min(lons),
                'max_lon': max(lons)
            }

        return bounds



def write_to_log(logging_this_data):
    log_location = f"{cwd}/GIS_Data/Change_Logs"
    with open(f"{log_location}/{day}_{hour}.txt", "a" ) as log:
        log.write(f"\n{logging_this_data}\n|\n---> at {day}{hour}{minute} to {log_location}/{day}_{hour}.txt\n")

#----------------------------------- CURRENT TEST -----------------------------------
def test_all_tiffs_in_folder(folder_path, grid_size=10):
    """
    Runs test_functionality_overlay() on every .tiff file in a folder.
    """
    print(f"\n=== Running overlay tests for folder: {folder_path} ===")

    if not os.path.isdir(folder_path):
        print(f"ERROR: Folder does not exist: {folder_path}")
        return

    tiff_files = glob.glob(os.path.join(folder_path, "*.tiff"))

    if not tiff_files:
        print("No .tiff files found in folder.")
        return

    print(f"Found {len(tiff_files)} TIFF files.\n")

    for idx, tiff_file in enumerate(tiff_files, 1):
        print(f"\n===============================\n"
              f"  Testing file {idx}/{len(tiff_files)}:\n  {tiff_file}\n"
              f"===============================")
        write_to_log(f"\n\n------------------------------------------------------------- \nTesting file {idx}/{len(tiff_files)}:\n  {tiff_file}")
              

        try:
            test_functionality_overlay(tiff_file, grid_size)
            print(f"SUCCESS → {os.path.basename(tiff_file)}")
        except Exception as e:
            print(f"ERROR while testing {tiff_file}: {e}")

    print("\n=== Completed testing all TIFFs ===")
    write_to_log(f"\n|\n--->Completed testing all TIFFs")




def test_functionality_overlay(tiff_file, grid_size=10):
    """
    Test all main Functionality class methods:
    1. get_latlon_bounds
    2. plot_latlon_on_tiff
    3. overlay_fire_predictions_on_tiff
    """
    print("===== Testing get_latlon_bounds =====")
    bounds = Functionality.get_latlon_bounds(tiff_file)
    print("Raster Lat/Lon Bounds:")
    print(f"Latitude: {bounds['min_lat']} → {bounds['max_lat']}")
    print(f"Longitude: {bounds['min_lon']} → {bounds['max_lon']}\n")

    # Pick a test point in the center of the raster bounds
    test_lat = (bounds['min_lat'] + bounds['max_lat']) / 2
    test_lon = (bounds['min_lon'] + bounds['max_lon']) / 2

    print("===== Testing plot_latlon_on_tiff =====")
    print(f"Plotting test point at Lat: {test_lat}, Lon: {test_lon}")
    fig_point = Functionality.plot_latlon_on_tiff(
        tiff_path=tiff_file,
        lat=test_lat,
        lon=test_lon,
        marker_color='blue',
        marker_size=100,
        title="Test Point on Raster"
    )
    print("Point plot complete!\n")
    write_to_log(f"\n|\n--->Point plot complete!\n")

    print("===== Testing overlay_fire_predictions_on_tiff =====")
    write_to_log("\n|\n--->Testing overlay_fire_predictions_on_tiff")
    # Generate dummy fire risk and spread maps
    fire_risk_map = np.random.rand(grid_size, grid_size)
    fire_spread_map = np.random.rand(grid_size, grid_size)

    overlay_folder = f"{cwd_safe}\GIS_Data\Opaque_Overlays"
    fig_overlay = Functionality.overlay_fire_predictions_on_tiff(
        tiff_path=tiff_file,
        fire_risk_map=fire_risk_map,
        fire_spread_map=fire_spread_map,
        alpha=0.5,
        output_folder=overlay_folder
    )
    print("Overlay plot complete!")

    # Optionally show figures
    # fig_point.show()
    # fig_overlay.show()

def test_functionality(tiff_file):
    """
    Test Functionality class methods:
    1. get_latlon_bounds
    2. plot_latlon_on_tiff
    """
    print("===== Testing get_latlon_bounds =====")
    bounds = Functionality.get_latlon_bounds(tiff_file)
    print("Raster Lat/Lon Bounds:")
    print(f"Latitude: {bounds['min_lat']} → {bounds['max_lat']}")
    print(f"Longitude: {bounds['min_lon']} → {bounds['max_lon']}\n")

    # Pick a test point in the center of the raster bounds
    test_lat = (bounds['min_lat'] + bounds['max_lat']) / 2
    test_lon = (bounds['min_lon'] + bounds['max_lon']) / 2

    print("===== Testing plot_latlon_on_tiff =====")
    print(f"Plotting test point at Lat: {test_lat}, Lon: {test_lon}")

    fig = Functionality.plot_latlon_on_tiff(
        tiff_path=tiff_file,
        lat=test_lat,
        lon=test_lon,
        marker_color='blue',
        marker_size=100,
        title="Test Point on Raster"
    )

    # Show the plot
    #fig.show()
    print("Test plot complete!")

def Test_Gen_Usage():
    weather_temp, weather_humidity = DataSim.generate_weather_data()
    
    fuel_loading = DataSim.generate_fuel_loading()
    def test_gen_display():
        tempcount = 0
        humidcount = 0
        fuelcount = 0
        print(f"\nGenerated Weather - Temperature:\n")
        for temp in weather_temp:
            tempcount += 1
            for temp2 in temp:
                print(f"temp value {tempcount}: {temp2}")
        print(f"\nGenerated Weather - Humidity:\n")
        for temp in weather_humidity:
            humidcount += 1
            for temp2 in temp:
                print(f"temp value {humidcount}: {temp2}")
        print(f"\nGenerated Terrain - Fuel Loading:\n")
        for temp in fuel_loading:
            fuelcount += 1
            for temp2 in temp:
                print(f"temp value {fuelcount}: {temp2}")
    #test_gen_display()

    #raw_file_storage = '/Users/adam/Documents/GitHub/ML_Fire_Prediction_Modeling/GIS_Data/raw_tiff'
    raw_file_storage = f"{cwd_safe}/GIS_Data/raw_tiff"
    #pwd_file_storage = os.
    raw_tiff_files = [f for f in os.listdir(raw_file_storage) if f.endswith('.tiff')]
    print(f"raw_tiff_files: {raw_tiff_files}\n")


def run_fire_spread_test():
    # 1. Generate random weather
    weather_temp, weather_humidity = DataSim.generate_weather_data()

    # 2. Generate random fuel loading
    fuel_loading = DataSim.generate_fuel_loading()

    # 3. Create a dummy elevation map and compute slope
    elevation = np.random.uniform(0, 100, (grid_size, grid_size))
    slope = FireSpreadModel.compute_slope(elevation)

    # 4. Combine all features into fire risk
    fire_spread_risk = FireSpreadModel.combine_features(
        weather_temp,
        weather_humidity,
        slope,
        fuel_loading
    )

    # 5. Choose a start location for fire (e.g., center of map)
    start_location = (grid_size // 2, grid_size // 2)

    # 6. Run fire spread simulation
    fire_map = FireSpreadModel.spread_fire(fire_spread_risk, start_location)

    print("\nFire Spread Map (1 = burned):\n")
    print(fire_map)

    # 7. Optionally visualize
    DataVisualization.plot_2d(fire_map, "Fire Spread Simulation", cmap="hot")

    return fire_map

def run_fire_spread_real_data(my_elevation_array, my_fuel_array, my_temp_array, my_humidity_array, grid_size=10):
    # Create the model instance
    model = FireSpreadModel(grid_size=grid_size)
    
    # Run fire spread simulation with your actual data
    fire_map = model.run_fire_spread(
        elevation=my_elevation_array,
        fuel_loading=my_fuel_array,
        weather_temp=my_temp_array,
        weather_humidity=my_humidity_array
    )
    
    return fire_map




def run_fire_simulation_for_all_tiffs(raw_folder, norm_folder, grid_size=10, start_location=None, outlier_threshold=2):
    """
    Run fire risk and spread simulation for all TIFFs in a folder.

    Args:
        raw_folder (str): Path to raw TIFFs.
        norm_folder (str): Path to save normalized TIFFs and outputs.
        grid_size (int): Grid size for downsampling/resampling.
        start_location (tuple): Fire start location (row, col). Defaults to center.
        outlier_threshold (float): Threshold for outlier removal.
    
    Returns:
        dict: Dictionary of results {tiff_name: (fire_risk_map, fire_spread_map)}
    """

    os.makedirs(norm_folder, exist_ok=True)

    # Normalize all TIFFs
    normalized_files = ObtainActualData.process_and_normalize_raw_tiffs(
        raw_folder=raw_folder,
        output_folder=norm_folder,
        grid_size=grid_size
    )

    print("Normalized TIFFs:", normalized_files)

    # Initialize model
    model = FireSpreadModel(grid_size=grid_size)

    # Set default start location if None
    if start_location is None:
        start_location = (grid_size // 2, grid_size // 2)

    results = {}

    # Run simulation for each normalized TIFF
    for tiff_name, tiff_path in normalized_files.items():
        print(f"\nRunning simulation for {tiff_name}...")

        fire_risk_map, fire_spread_map = model.run_fire_spread(
            elevation_file=tiff_path,
            fuel_file=None,
            temp_file=None,
            humidity_file=None,
            start_location=start_location,
            outlier_threshold=outlier_threshold
        )

        # Save outputs per file
        base_name = os.path.splitext(tiff_name)[0]

        fire_risk_npy_saving = f"{cwd_safe}/GIS_Data/Risk_Prediction_Output/NPY_Data_Type/{base_name}_Predictions"
        
        fire_spread_npy_saving = f"{cwd_safe}/GIS_Data/Spread_Prediction_Output/NPY_Data_Type/{base_name}_Predictions"

        os.makedirs(f"{fire_risk_npy_saving}", exist_ok=True)
        os.makedirs(f"{fire_spread_npy_saving}", exist_ok=True)

        np.save(f"{fire_risk_npy_saving}/{base_name}_fire_risk.npy", fire_risk_map)
        np.save(f"{fire_spread_npy_saving}/{base_name}_fire_spread.npy", fire_spread_map)

        print(f"Saved fire risk and spread maps for {tiff_name}")

        # Store in results dictionary
        results[tiff_name] = (fire_risk_map, fire_spread_map)

    return results





# ------------------- Directories -------------------
raw_dir = f"{cwd_safe}/GIS_Data/raw_tiff"
norm_dir = f"{cwd_safe}/GIS_Data/normalized_tiff"

# Example usage:
tiff_test_file = f"{cwd_safe}/GIS_Data/raw_tiff/Conifer.tiff"
#test_functionality_overlay(tiff_test_file)


test_all_tiffs_in_folder(norm_dir)
test_all_tiffs_in_folder(raw_dir)



# Run everything in one call

#all_results = run_fire_simulation_for_all_tiffs(
#    raw_folder=raw_dir,
#    norm_folder=norm_dir,
#    grid_size=10
#)





print("\nSimulation complete for all TIFFs.")
