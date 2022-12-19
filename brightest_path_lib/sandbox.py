from algorithm import AStarSearch
import time
from skimage import data
import numpy as np
import napari

def test_2D_image():
    # testing for 2D
    #twoDImage = data.cells3d()[30, 1]  # brighter image
    twoDImage = data.cells3d()[30, 0] # darker image
    start_point = np.array([0,192])
    goal_point = np.array([198,9])

    astar_search = AStarSearch(
        image=twoDImage,
        start_point=start_point,
        goal_point=goal_point)
    tic = time.perf_counter()
    result = astar_search.search()
    toc = time.perf_counter()
    print(f"Found brightest path in {toc - tic:0.4f} seconds")
    print(f"path size: {len(result)}")

    viewer = napari.Viewer()
    viewer.add_image(twoDImage[:200, :200], colormap='magma')
    viewer.add_points(np.array([start_point, goal_point]), size=2, edge_width=1, face_color="white", edge_color="white")
    viewer.add_points(result, size=1, edge_width=1, face_color="green", edge_color="green")
    napari.run()

def test_3D_image():
    # testing for 3D
    threeDImage = data.cells3d()[30]  # grab some data
    astar_search = AStarSearch(
        image=threeDImage,
        start_point=np.array([1, 16, 1]),
        goal_point=np.array([1, 35, 1]),
        )

    result = astar_search.search()
    print(result)

    viewer = napari.Viewer()
    viewer.add_image(threeDImage[:200, :200], colormap='magma')
    viewer.add_points(result, size=1, edge_width=1, face_color="green", edge_color="green")
    napari.run()

if __name__ == "__main__":
    test_2D_image()
