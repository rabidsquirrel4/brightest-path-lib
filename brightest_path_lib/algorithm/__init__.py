# algorithm/__init__.py

"""Find the brightest path between two points.

Modules exported by this package:

- `AStarSearch`: This module allows you to search for the brightest path using the A* Search algorithm
- `NBAStarSearch`: This module allows you to search for the brightest path using a Bidirectional Search algorithm called NBA*
"""

from .astar import AStarSearch
from .nbastar import NBAStarSearch