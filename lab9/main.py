import pathlib
import os
import geopandas
import pandas
import matplotlib.pyplot
import shapely.geometry
import itertools

SCRIPT_PATH = pathlib.Path(os.path.abspath(__file__))
SCRIPT_DIR = SCRIPT_PATH.parent
BUILD_DIR = SCRIPT_DIR / 'build'

def build_distance_grid(gdf: geopandas.GeoDataFrame, num_stations: int) -> tuple[geopandas.GeoDataFrame, float, geopandas.GeoDataFrame]:
    sample_gdf = gdf.head(num_stations).copy()
    
    projected_gdf = sample_gdf.to_crs("EPSG:3857")
    assert projected_gdf.crs.to_epsg() == 3857
    
    lines_geometry = []
    for point1, point2 in itertools.combinations(projected_gdf.geometry, 2):
        lines_geometry.append(shapely.geometry.LineString([point1, point2]))
        
    grid_gdf = geopandas.GeoDataFrame({'geometry': lines_geometry}, crs=projected_gdf.crs)
    
    distances = grid_gdf.geometry.length
    average_distance_meters = distances.mean()
    
    return grid_gdf, average_distance_meters, projected_gdf

def calculate_density_zones(gdf: geopandas.GeoDataFrame, grid_x: int, grid_y: int) -> tuple[geopandas.GeoDataFrame, geopandas.GeoSeries]:
    min_x, min_y, max_x, max_y = gdf.total_bounds
    step_x = (max_x - min_x) / grid_x
    step_y = (max_y - min_y) / grid_y
    
    polygon_cells = []
    for i in range(grid_x):
        for j in range(grid_y):
            x_start = min_x + i * step_x
            y_start = min_y + j * step_y
            x_end = x_start + step_x
            y_end = y_start + step_y
            polygon_cells.append(shapely.geometry.box(x_start, y_start, x_end, y_end))
            
    spatial_grid = geopandas.GeoDataFrame({'geometry': polygon_cells}, crs=gdf.crs)
    
    joined_data = geopandas.sjoin(gdf, spatial_grid, how='left', predicate='within')
    station_counts = joined_data['index_right'].value_counts()
    
    top_indices = station_counts.head(5).index
    top_zones = spatial_grid.iloc[top_indices].copy()
    zone_centroids = top_zones.geometry.centroid
    
    return top_zones, zone_centroids

def generate_distance_visualization(grid_gdf: geopandas.GeoDataFrame, points_gdf: geopandas.GeoDataFrame, base_gdf: geopandas.GeoDataFrame, avg_dist: float) -> None:
    figure, axis = matplotlib.pyplot.subplots(figsize=(16, 10))
    
    projected_base = base_gdf.to_crs(grid_gdf.crs)
    projected_base.plot(ax=axis, color='lightgray', markersize=1, alpha=0.3, zorder=1)
    
    grid_gdf.plot(ax=axis, color='blue', linewidth=0.3, alpha=0.6, zorder=2)
    points_gdf.plot(ax=axis, color='red', markersize=15, zorder=3)
    
    axis.set_title(f"Distance Grid Network\nAverage Distance: {avg_dist / 1000:.2f} km")
    axis.set_axis_off()
    
    output_path = BUILD_DIR / "distance_grid_network.png"
    matplotlib.pyplot.savefig(output_path, format="png", bbox_inches='tight')
    output_path = BUILD_DIR / "distance_grid_network.svg"
    matplotlib.pyplot.savefig(output_path, format="svg", bbox_inches='tight')

    matplotlib.pyplot.close(figure)

def generate_density_visualization(base_gdf: geopandas.GeoDataFrame, zones_gdf: geopandas.GeoDataFrame, centroids: geopandas.GeoSeries) -> None:
    figure, axis = matplotlib.pyplot.subplots(figsize=(16, 10))
    
    base_gdf.plot(ax=axis, color='gray', markersize=1, alpha=0.3, zorder=1)
    zones_gdf.boundary.plot(ax=axis, color='red', linewidth=2.5, zorder=2)
    centroids.plot(ax=axis, color='darkblue', marker='*', markersize=200, zorder=3)
    
    axis.set_title(f"Top {len(centroids)} Highest Density Areas and Their Centroids")
    axis.set_axis_off()
    
    output_path = BUILD_DIR / "highest_density_zones.png"
    matplotlib.pyplot.savefig(output_path, format="png", bbox_inches='tight')
    output_path = BUILD_DIR / "highest_density_zones.svg"
    matplotlib.pyplot.savefig(output_path, format="svg", bbox_inches='tight')
    matplotlib.pyplot.close(figure)

def main() -> None:
    os.makedirs(BUILD_DIR, exist_ok=True)
    
    shapefile_path = SCRIPT_DIR / "Lab_work_9_example" / "GIS_distance_example" / "Fire_Stations" / "Fire_Stations.shp"
    assert shapefile_path.exists()
    
    fire_stations_data = geopandas.read_file(shapefile_path)
    
    distance_grid, avg_distance, projected_points = build_distance_grid(fire_stations_data, 40)
    generate_distance_visualization(distance_grid, projected_points, fire_stations_data, avg_distance)
    
    density_zones, zone_centroids = calculate_density_zones(fire_stations_data, 50, 40)
    generate_density_visualization(fire_stations_data, density_zones, zone_centroids)

if __name__ == '__main__':
    main()