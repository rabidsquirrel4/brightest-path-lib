import sys
sys.path.append("/Users/vasudhajha/Documents/mapmanager/brightest-path-lib")

from brightest_path_lib.algorithm import AStarSearch, NBAStarSearch
import numpy as np
import napari
from skimage import data
import tifffile
import time


def test_2D_image():
    # testing for 2D
    #twoDImage = data.cells3d()[30, 1]  # brighter image
    # twoDImage = data.cells3d()[30, 0] # darker image
    # start_point = np.array([0,192])
    # goal_point = np.array([198,9])

    image = tifffile.imread('rr30a_s0_ch2.tif')[30]
    start_point = np.array([243, 292]) # (y,x)
    goal_point = np.array([128, 711]) # (y,x)

    astar_search = AStarSearch(
        image,
        start_point,
        goal_point)
    tic = time.perf_counter()
    result = astar_search.search()
    toc = time.perf_counter()
    print(f"Found brightest path in {toc - tic:0.4f} seconds")
    print(f"path size: {len(result)}")
    print(f"Number of nodes viewed: {astar_search.evaluated_nodes}")

    # nbastar_search = NBAStarSearch(
    #     image,
    #     start_point,
    #     goal_point)
    # tic = time.perf_counter()
    # result = nbastar_search.search()
    # toc = time.perf_counter()
    # print(f"Found brightest path in {toc - tic:0.4f} seconds")
    # print(f"path size: {len(result)}")
    #print(f"Number of nodes viewed: {nbastar_search.evaluated_nodes}")

    viewer = napari.Viewer()
    # viewer.add_image(twoDImage[:100, :250], colormap='magma')
    viewer.add_image(image)
    viewer.add_points(np.array([start_point, goal_point]), size=10, edge_width=1, face_color="red", edge_color="red")
    viewer.add_points(result, size=10, edge_width=1, face_color="green", edge_color="green")
    napari.run()

def test_3D_image():
    image = tifffile.imread('rr30a_s0_ch2.tif')
    start_point = np.array([30, 243, 292]) # (z,y,x)
    goal_point = np.array([30, 221, 434]) # (z,y,x)

    astar_search = AStarSearch(
        image,
        start_point,
        goal_point
    )

    tic = time.perf_counter()
    result = astar_search.search()
    toc = time.perf_counter()
    print(f"Found brightest path in {toc - tic:0.4f} seconds")
    print(f"path size: {len(result)}")

    # nbastar_search = NBAStarSearch(
    #     image,
    #     start_point,
    #     goal_point
    # )
    # tic = time.perf_counter()
    # result = nbastar_search.search()
    # toc = time.perf_counter()
    # print(f"Found brightest path in {toc - tic:0.4f} seconds")
    # print(f"path size: {len(result)}")

    viewer = napari.Viewer()
    viewer.add_image(image)
    viewer.add_points([start_point, goal_point], size=10, edge_width=1, face_color="red", edge_color="red")
    viewer.add_points(result, size=10, edge_width=1, face_color="green", edge_color="green")
    napari.run()

if __name__ == "__main__":
    test_2D_image()
