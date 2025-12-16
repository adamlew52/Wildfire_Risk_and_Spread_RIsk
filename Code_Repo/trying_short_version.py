import os
import re
import glob
import numpy as np
import rasterio
from rasterio.plot import show
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from pyproj import Transformer
from datetime import datetime

# ------------------- CONFIG -------------------
GRID_SIZE = 10
OUTLIER_THRESHOLD = 2
cwd = os.getcwd()
cwd_safe = re.sub(r'[<>"/\\|?* ]', '/', cwd)

OUTPUT_DIR = f"{cwd_safe}/GIS_Data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

now = datetime.now()
DATE_STR = f"{now.day}_{now.month}_{now.year}_{now.hour}_{now.minute}"

# ------------------- DATA SIMULATION -------------------
class DataSim:
    @staticmethod
    def generate_weather(grid_size=GRID_SIZE):
        temp = np.random.uniform(30, 40, (grid_size, grid_size))
        humidity = np.random.uniform(10, 30, (grid_size, grid_size))
        return temp, humidity

    @staticmethod
    def generate_fuel(grid_size=GRID_SIZE):
        return np.random.uniform(0, 1, (grid_size, grid_size))


# ------------------- GIS DATA HANDLING -------------------
class GISUtils:
    @staticmethod
    def load_tiff(file_path):
        with rasterio.open(file_path) as src:
            data = src.read(1).astype("float32")
            return data, src.transform, src.crs

    @staticmethod
    def normalize(data):
        data = data.astype("float32")
        data[data == -9999] = np.nan
        min_val, max_val = np.nanmin(data), np.nanmax(data)
        if max_val == min_val:
            return np.zeros_like(data)
        return (data - min_val) / (max_val - min_val)

    @staticmethod
    def resample(data, target_size):
        zoom_x = target_size / data.shape[0]
        zoom_y = target_size / data.shape[1]
        return zoom(data, (zoom_x, zoom_y), order=1)

    @staticmethod
    def remove_outliers(data, threshold=OUTLIER_THRESHOLD):
        mean, std = np.mean(data), np.std(data)
        return np.clip(data, mean - threshold * std, mean + threshold * std)

    @staticmethod
    def save_tiff(path, data, crs, transform):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with rasterio.open(path, "w", driver="GTiff",
                           height=data.shape[0], width=data.shape[1],
                           count=1, dtype="float32", crs=crs, transform=transform) as dst:
            dst.write(data, 1)

    @staticmethod
    def latlon_to_rowcol(file_path, lat, lon):
        with rasterio.open(file_path) as src:
            transform, crs = src.transform, src.crs
            if crs.is_geographic:
                x, y = lon, lat
            else:
                transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
                x, y = transformer.transform(lon, lat)
            col, row = ~transform * (x, y)
        return int(round(row)), int(round(col))

    @staticmethod
    def get_bounds(file_path):
        with rasterio.open(file_path) as src:
            corners = [(0, 0), (src.width, 0), (0, src.height), (src.width, src.height)]
            map_coords = [src.transform * (c, r) for c, r in corners]
            if src.crs.is_geographic:
                lons, lats = zip(*map_coords)
            else:
                transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
                lons, lats = zip(*[transformer.transform(x, y) for x, y in map_coords])
        return {'min_lat': min(lats), 'max_lat': max(lats), 'min_lon': min(lons), 'max_lon': max(lons)}


# ------------------- FIRE MODEL -------------------
class FireSpreadModel:
    def __init__(self, grid_size=GRID_SIZE):
        self.grid_size = grid_size

    def normalize(self, data):
        min_val, max_val = np.min(data), np.max(data)
        return (data - min_val) / (max_val - min_val) if max_val != min_val else np.zeros_like(data)

    def compute_slope(self, elevation):
        grad = np.gradient(elevation)
        slope = np.sqrt(grad[0] ** 2 + grad[1] ** 2)
        return self.normalize(slope)

    def combine_features(self, temp, humidity, slope, fuel):
        risk = (self.normalize(temp) * 0.5 +
                (1 - self.normalize(humidity)) * 0.3 +
                self.normalize(slope) * 0.5 +
                self.normalize(fuel) * 0.3)
        return self.normalize(risk)

    def spread_fire(self, fire_risk, start=(GRID_SIZE//2, GRID_SIZE//2), decay=0.7, steps=5):
        fire_map = np.zeros_like(fire_risk, dtype=float)
        fire_map[start] = fire_risk[start]
        for _ in range(steps):
            new_map = fire_map.copy()
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if fire_map[i,j] > 0:
                        neighbors = [(i+di, j+dj) for di in (-1,0,1) for dj in (-1,0,1)
                                     if 0 <= i+di < self.grid_size and 0 <= j+dj < self.grid_size]
                        for ni,nj in neighbors:
                            new_map[ni,nj] = max(new_map[ni,nj], fire_map[i,j]*fire_risk[ni,nj]*decay)
            fire_map = new_map
        return fire_map / fire_map.max() if fire_map.max() != 0 else fire_map

    def run(self, elevation=None, fuel=None, temp=None, humidity=None, start_location=None):
        elevation = elevation if elevation is not None else np.random.uniform(0, 100, (self.grid_size, self.grid_size))
        fuel = fuel if fuel is not None else DataSim.generate_fuel(self.grid_size)
        temp, humidity = (temp, humidity) if temp is not None and humidity is not None else DataSim.generate_weather(self.grid_size)

        # Outlier removal
        elevation = GISUtils.remove_outliers(elevation)
        fuel = GISUtils.remove_outliers(fuel)
        temp = GISUtils.remove_outliers(temp)
        humidity = GISUtils.remove_outliers(humidity)

        slope = self.compute_slope(elevation)
        fire_risk = self.combine_features(temp, humidity, slope, fuel)

        if start_location is None:
            start_location = (self.grid_size//2, self.grid_size//2)

        fire_spread = self.spread_fire(fire_risk, start=start_location)
        return fire_risk, fire_spread


# ------------------- VISUALIZATION -------------------
class Visualization:
    @staticmethod
    def plot_2d(data, title="", cmap="hot"):
        fig, ax = plt.subplots(figsize=(8,8))
        cax = ax.imshow(data, origin='lower', cmap=cmap)
        fig.colorbar(cax, ax=ax)
        ax.set_title(title)
        return fig


# ------------------- TESTING / UTILITY -------------------
def write_log(message):
    log_dir = f"{OUTPUT_DIR}/Change_Logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/{DATE_STR}.txt"
    with open(log_file, "a") as f:
        f.write(f"{message}\n")

def test_single_tiff(tiff_path, grid_size=GRID_SIZE, ignition_lat=None, ignition_lon=None):
    print(f"Testing TIFF: {tiff_path}")
    bounds = GISUtils.get_bounds(tiff_path)
    if ignition_lat is None or ignition_lon is None:
        ignition_lat = (bounds['min_lat'] + bounds['max_lat'])/2
        ignition_lon = (bounds['min_lon'] + bounds['max_lon'])/2

    row, col = GISUtils.latlon_to_rowcol(tiff_path, ignition_lat, ignition_lon)
    row_scaled = min(grid_size-1, max(0, int(row / (row / grid_size))))
    col_scaled = min(grid_size-1, max(0, int(col / (col / grid_size))))

    model = FireSpreadModel(grid_size)
    fire_risk = np.random.rand(grid_size, grid_size)
    fire_spread = np.zeros((grid_size, grid_size))
    fire_spread[row_scaled, col_scaled] = 1.0

    overlay_folder = f"{OUTPUT_DIR}/Overlays"
    os.makedirs(overlay_folder, exist_ok=True)
    fig = Visualization.plot_2d(fire_spread, title="Test Overlay")
    fig.savefig(f"{overlay_folder}/overlay_{DATE_STR}.pdf")
    print("Test overlay saved.")


def run_all_tiffs(folder_path, grid_size=GRID_SIZE):
    tiffs = glob.glob(os.path.join(folder_path, "*.tiff"))
    for tiff in tiffs:
        test_single_tiff(tiff, grid_size=grid_size)


# ------------------- DIRECTORIES -------------------
RAW_DIR = f"{OUTPUT_DIR}/raw_tiff"
NORM_DIR = f"{OUTPUT_DIR}/normalized_tiff"

# ------------------- EXAMPLE USAGE -------------------
# test all TIFFs in a folder
run_all_tiffs(NORM_DIR)
