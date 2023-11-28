'''
Proof of concept of A-star pathfinding algorithm 
We use this implementation: https://github.com/mapmanager/brightest-path-lib
See here for more details, this is recent: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10370081/
Author: Lucien Werner
Date: Nov 13 2023
Edited by: Rachel Shi
Date: Nov 27 2023
'''

# 3rd-party Imports
import numpy as np
import matplotlib.pyplot as plt
from brightest_path_lib.algorithm import AStarSearch # type: ignore
from brightest_path_lib.input import CostFunction, HeuristicFunction # type: ignore
import matplotlib.image as mpimg
import cv2
import time


# test image
# height, width = 101, 101
# im = np.zeros((height, width))
# start_pixel, end_pixel = (50,10), (50,90)  # row, column
# im[50,10:90] = 1

file = "images/substation3_200.png"
# get image with rgb
img_rgba = mpimg.imread(file)
img_gray = 1 - cv2.cvtColor(img_rgba, cv2.COLOR_BGR2GRAY)  # invert color
print(f"Image Shape: {img_gray.shape}")


# hand code the start and end pixels for now to explore file resolutions
# start_end_dict = {"src/images/substation3_clip.png": [(37, 46), (679, 125)],
#                   "src/images/substation3_50.png": [(564, 1061), (1278, 1150)],
#                   "src/images/substation3_100.png": [(1108, 2122), (2558, 2300)],
#                   "src/images/substation3_200.png": [(2220, 4244), (5116, 4601)],
#                   }

# dictionary of piexels for objects for 200 image
start_end_dict = {"pair1": [(2220, 4244), (5116, 4601)],
                  "pair2": [(328, 1137), (702, 7413)],
                  "pair3": [(2955, 1571), (5171, 3577)],
                  "pair4": [(4171, 5742), (5516, 5860)],
                  "pair5": [(5110, 940), (2991, 1217)],
                  }

pair = "pair4"
start_pixel, end_pixel = start_end_dict[pair][0], start_end_dict[pair][1]


# TODO: change to reflect new astar search changes
# consistency_cost_func: CostFunction = ReciprocalConsistency()
# sand_trap_heuristic_func: HeuristicFunction = 
# astar = AStarSearch(img, start_pixel, end_pixel, 
#                     cost_function=consistency_cost_func, 
#                     heuristic_function=sand_trap_heuristic_func)
# path = astar.search()
astar = AStarSearch(img_gray, img_rgba, start_pixel, end_pixel)
astar_start_time = time.time()
print("Starting AStar ...")
path = astar.search()
astar_end_time = time.time()
print("AStar ended.")
print(f"Astar Run Time: " + 
        f"{astar_end_time - astar_start_time} seconds")

plt.imshow(img_rgba)
plt.plot(start_pixel[1], start_pixel[0], 'og')
plt.plot(end_pixel[1], end_pixel[0], 'or')
plt.plot([point[1] for point in astar.result], [point[0] for point in path], '-b', linewidth=3)
plt.plot(start_pixel[1], start_pixel[0], 'og')
plt.plot(end_pixel[1], end_pixel[0], 'or')
plt.tight_layout()
plt.savefig(f"images/substation3_200_{pair}.png")
# plt.show()
plt.close()

print('Done')