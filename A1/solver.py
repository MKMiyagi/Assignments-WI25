import math
import random
from collections import deque, defaultdict
import heapq
import numpy as np

random.seed(42)

###############################################################################
#                                Node Class                                   #
###############################################################################

class Node:
    """
    Represents a graph node with an undirected adjacency list.
    'value' can store (row, col), or any unique identifier.
    'neighbors' is a list of connected Node objects (undirected).
    """
    def __init__(self, value):
        self.value = value
        self.neighbors = []

    def add_neighbor(self, node):
        """
        Adds an undirected edge between self and node:
         - self includes node in self.neighbors
         - node includes self in node.neighbors (undirected)
        """
        # TODO: Implement adding a neighbor in an undirected manner
        self.neighbors.append(node)  

    def __repr__(self):
        return f"Node({self.value})"
    
    def __lt__(self, other):
        return self.value < other.value


###############################################################################
#                   Maze -> Graph Conversion (Undirected)                     #
###############################################################################

def parse_maze_to_graph(maze):
    """
    Converts a 2D maze (numpy array) into an undirected graph of Node objects.
    maze[r][c] == 0 means open cell; 1 means wall/blocked.

    Returns:
        nodes_dict: dict[(r, c): Node] mapping each open cell to its Node
        start_node : Node corresponding to (0, 0), or None if blocked
        goal_node  : Node corresponding to (rows-1, cols-1), or None if blocked
    """
    rows, cols = maze.shape
    nodes_dict = {}

    # 1) Create a Node for each open cell
    # 2) Link each node with valid neighbors in four directions (undirected)
    # 3) Identify start_node (if (0,0) is open) and goal_node (if (rows-1, cols-1) is open)

    # TODO: Implement the logic to build nodes and link neighbors

    if maze.size == 0:
        return None, None, None

    for row in range(rows):
        for col in range(cols):
            if maze[row][col] == 0:
                nodes_dict[(row, col)] = Node((row, col))
    
    for row, col in nodes_dict.keys():
        if row < rows - 1 and (row + 1, col) in nodes_dict:
            nodes_dict[(row, col)].add_neighbor(nodes_dict[(row + 1, col)])
            nodes_dict[(row + 1, col)].add_neighbor(nodes_dict[(row, col)])
        
        if row > 0 and (row - 1, col) in nodes_dict:
            nodes_dict[(row, col)].add_neighbor(nodes_dict[(row - 1, col)])
            nodes_dict[(row - 1, col)].add_neighbor(nodes_dict[(row, col)])

        if col > 0 and (row, col - 1) in nodes_dict:
            nodes_dict[(row, col)].add_neighbor(nodes_dict[(row, col - 1)])
            nodes_dict[(row, col - 1)].add_neighbor(nodes_dict[(row, col)])

        if col < cols - 1 and (row, col + 1) in nodes_dict:
            nodes_dict[(row, col)].add_neighbor(nodes_dict[(row, col + 1)])
            nodes_dict[(row, col + 1)].add_neighbor(nodes_dict[(row, col)])

    # TODO: Assign start_node and goal_node if they exist in nodes_dict
    start_node = nodes_dict.get((0, 0), None)
    goal_node = nodes_dict.get((rows - 1, cols - 1), None)

    return nodes_dict, start_node, goal_node


###############################################################################
#                         BFS (Graph-based)                                    #
###############################################################################

def bfs(start_node, goal_node):
    """
    Breadth-first search on an undirected graph of Node objects.
    Returns a list of (row, col) from start to goal, or None if no path.

    Steps (suggested):
      1. Use a queue (collections.deque) to hold nodes to explore.
      2. Track visited nodes so you don’t revisit.
      3. Also track parent_map to reconstruct the path once goal_node is reached.
    """
    # TODO: Implement BFS
    if not start_node or not goal_node:
        return None

    visited = []
    queue = [start_node]
    parent_map = {start_node: None}

    while queue:
        node = queue.pop(0)
        visited.append(node)

        if node == goal_node:
            break

        for neighbor in node.neighbors:
            if neighbor not in visited:
                visited.append(neighbor)
                if neighbor in queue:
                    continue
                else:
                    queue.append(neighbor)
                    parent_map[neighbor] = node
    
    if goal_node not in parent_map:
        return None
    
    return reconstruct_path(goal_node, parent_map)


###############################################################################
#                          DFS (Graph-based)                                   #
###############################################################################

def dfs(start_node, goal_node):
    """
    Depth-first search on an undirected graph of Node objects.
    Returns a list of (row, col) from start to goal, or None if no path.

    Steps (suggested):
      1. Use a stack (Python list) to hold nodes to explore.
      2. Keep track of visited nodes to avoid cycles.
      3. Reconstruct path via parent_map if goal_node is found.
    """
    # TODO: Implement DFS
    if not start_node or not goal_node:
        return None
    
    visited = []
    stack = [start_node]
    parent_map = {start_node: None}

    while stack:
        node = stack.pop()
        visited.append(node)

        if node == goal_node:
            break

        for neighbor in node.neighbors:
            if neighbor not in visited:
                visited.append(neighbor)
                if neighbor in stack:
                    continue
                else:
                    stack.append(neighbor)
                    parent_map[neighbor] = node
    
    if goal_node not in parent_map:
        return None
    
    return reconstruct_path(goal_node, parent_map)

###############################################################################
#                    A* (Graph-based with Manhattan)                           #
###############################################################################

def astar(start_node, goal_node):
    """
    A* search on an undirected graph of Node objects.
    Uses manhattan_distance as the heuristic, assuming node.value = (row, col).
    Returns a path (list of (row, col)) or None if not found.

    Steps (suggested):
      1. Maintain a min-heap/priority queue (heapq) where each entry is (f_score, node).
      2. f_score[node] = g_score[node] + heuristic(node, goal_node).
      3. g_score[node] is the cost from start_node to node.
      4. Expand the node with the smallest f_score, update neighbors if a better path is found.
    """
    # TODO: Implement A*
    if not start_node or not goal_node:
        return None
    
    open_set = []
    parent_map = {start_node: None}
    visited = []
    heapq.heappush(open_set, (0, start_node))
    g_score = {start_node: 0}
    f_score = {start_node: manhattan_distance(start_node, goal_node)}

    while open_set:
        _, node = heapq.heappop(open_set)

        if node == goal_node:
            return reconstruct_path(goal_node, parent_map)
        
        visited.append(node)

        for neighbor in node.neighbors:
            if neighbor in visited:
                continue

            g_score_2 = g_score[node] + 1

            if neighbor not in g_score or g_score_2 < g_score[neighbor]:
                parent_map[neighbor] = node
                g_score[neighbor] = g_score_2
                f_score[neighbor] = g_score[neighbor] + manhattan_distance(neighbor, goal_node)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None

def manhattan_distance(node_a, node_b):
    """
    Helper: Manhattan distance between node_a.value and node_b.value 
    if they are (row, col) pairs.
    """
    # TODO: Return |r1 - r2| + |c1 - c2|
    r1, c1 = node_a.value
    r2, c2 = node_b.value

    return abs(r1-r2) + abs(c1-c2)


###############################################################################
#                 Bidirectional Search (Graph-based)                          #
###############################################################################

def bidirectional_search(start_node, goal_node):
    """
    Bidirectional search on an undirected graph of Node objects.
    Returns list of (row, col) from start to goal, or None if not found.

    Steps (suggested):
      1. Maintain two frontiers (queues), one from start_node, one from goal_node.
      2. Alternate expansions between these two queues.
      3. If the frontiers intersect, reconstruct the path by combining partial paths.
    """
    # TODO: Implement bidirectional search
    if not start_node or not goal_node:
        return None
    
    f_queue = [start_node]
    b_queue = [goal_node]
    f_visited = [start_node]
    b_visited = [goal_node]
    f_parent_map = {start_node: None}
    b_parent_map = {goal_node: None}

    while f_queue and b_queue:
        if f_queue:
            f_node = f_queue.pop(0)
            for neighbor in f_node.neighbors:
                if neighbor not in f_visited:
                    f_visited.append(neighbor)
                    f_parent_map[neighbor] = f_node
                    f_queue.append(neighbor)

                    if neighbor in b_visited:
                        return reconstruct_path_bd(neighbor, f_parent_map, b_parent_map)
        
        if b_queue:
            b_node = b_queue.pop(0)
            for neighbor in b_node.neighbors:
                if neighbor not in b_visited:
                    b_visited.append(neighbor)
                    b_parent_map[neighbor] = b_node
                    b_queue.append(neighbor)

                if neighbor in f_visited:
                    return reconstruct_path_bd(neighbor, f_parent_map, b_parent_map)
    return None

def reconstruct_path_bd(m_node, f_parent_map, b_parent_map):
    path = []
    node = m_node
    while node is not None:
        path.append(node.value)
        node = f_parent_map[node]
    
    path.reverse()

    node = b_parent_map[m_node]
    while node is not None:
        path.append(node.value)
        node = b_parent_map[node]
    
    return path


###############################################################################
#             Simulated Annealing (Graph-based)                               #
###############################################################################

def simulated_annealing(start_node, goal_node, temperature=1.0, cooling_rate=0.99, min_temperature=0.01):
    """
    A basic simulated annealing approach on an undirected graph of Node objects.
    - The 'cost' is the manhattan_distance to the goal.
    - We randomly choose a neighbor and possibly move there.
    Returns a list of (row, col) from start to goal (the path traveled), or None if not reached.

    Steps (suggested):
      1. Start with 'current' = start_node, compute cost = manhattan_distance(current, goal_node).
      2. Pick a random neighbor. Compute next_cost.
      3. If next_cost < current_cost, move. Otherwise, move with probability e^(-cost_diff / temperature).
      4. Decrease temperature each step by cooling_rate until below min_temperature or we reach goal_node.
    """
    # TODO: Implement simulated annealing
    if not start_node or not goal_node:
        return None

    node = start_node
    cost = manhattan_distance(node, goal_node)
    path = [node.value]

    while temperature > min_temperature:
        if node == goal_node:
            return path
        
        if not node.neighbors:
            break

        f_node = random.choice(node.neighbors)
        f_cost = manhattan_distance(f_node, goal_node)

        cost_d = f_cost - cost

        if cost_d < 0 or random.uniform(0, 1) < math.exp(-cost_d / temperature):
            node = f_node
            cost = f_cost
            path.append(node.value)
        
        temperature *= cooling_rate
    
    if node == goal_node:
        return path
    else:
        return None


###############################################################################
#                           Helper: Reconstruct Path                           #
###############################################################################

def reconstruct_path(end_node, parent_map):
    """
    Reconstructs a path by tracing parent_map up to None.
    Returns a list of node.value from the start to 'end_node'.

    'parent_map' is typically dict[Node, Node], where parent_map[node] = parent.

    Steps (suggested):
      1. Start with end_node, follow parent_map[node] until None.
      2. Collect node.value, reverse the list, return it.
    """
    # TODO: Implement path reconstruction
    path = []
    node = end_node
    while node in parent_map:
        path.append(node.value)
        node = parent_map[node]
    path.reverse()

    return path


###############################################################################
#                              Demo / Testing                                 #
###############################################################################
if __name__ == "__main__":
    # A small demonstration that the code runs (with placeholders).
    # This won't do much yet, as everything is unimplemented.
    random.seed(42)
    np.random.seed(42)

    # Example small maze: 0 => open, 1 => wall
    maze_data = np.array([
        [0, 0, 1],
        [0, 0, 0],
        [1, 0, 0]
    ])

    # Parse into an undirected graph
    nodes_dict, start_node, goal_node = parse_maze_to_graph(maze_data)
    print("Created graph with", len(nodes_dict), "nodes.")
    print("Start Node:", start_node)
    print("Goal Node :", goal_node)

    # Test BFS (will return None until implemented)
    path_bfs = bfs(start_node, goal_node)
    print("BFS Path:", path_bfs)
    path_dfs = bfs(start_node, goal_node)
    print("DFS Path:", path_dfs)
    path_astar = astar(start_node, goal_node)
    print("A* Path:", path_astar)
    path_bidirectional = bidirectional_search(start_node, goal_node)
    print("BDS Path:", path_bidirectional)
    path_sa = simulated_annealing(start_node, goal_node)
    print("SA Path:", path_sa)