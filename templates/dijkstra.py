import numpy as np
import time

nodes = []

class node:
    def __init__(self, symbol):
        self.symbol = symbol
        self.edges = []
        self.shortest_distance = float('inf')
        self.shortest_path_via = None

        nodes.append(self)

    def add_edge(self, node, distance):
        edge = [node, distance]
        if not edge in self.edges:
            self.edges.append(edge)

    def update_edges(self):
        for edge in self.edges:
            distance_via = self.shortest_distance + edge[1]
            if distance_via < edge[0].shortest_distance:
                edge[0].shortest_distance = distance_via
                edge[0].shortest_path_via = self

# Couples two nodes
def make_edge(node1, node2, distance):
    node1.add_edge(node2, distance)
    # node2.add_edge(node1, distance)

# Does the heavy lifting
# Just a python implementation of dijkstras shortest path
def dijkstra(start, end):
    global nodes
    queue = []
    path = []

    queue = nodes.copy()
    start.shortest_distance = 0
    queue.sort(key=lambda node: node.shortest_distance)

    while queue[0] != end:
        node = queue[0]
        node.update_edges()
        path.append(queue.pop(0))
        queue.sort(key=lambda node: node.shortest_distance)
    
    print(print_path(end))
    print(f"Path array: {get_path_array(end)}")
    print(f"Distance: {end.shortest_distance}")

# Literally just prints the path
def print_path(node):
    if node.shortest_path_via == None:
        return f"{node.symbol}"
    else:
        return f"{print_path(node.shortest_path_via)} -> {node.symbol}"

# Does what it says on the tin
def get_node(symbol):
    for node in nodes:
        if node.symbol == symbol:
            return node
    return 0

# Takes a set of edges, as well as start and end nodes
def solve_dijkstra(edges, start, end):
    # Make edges into nodes and couple them
    global nodes
    nodes = []
    for edge in edges:
        a = get_node(edge[0])
        b = get_node(edge[1])

        if a == 0:
            a = node(edge[0])

        if b == 0:
            b = node(edge[1])

        a.add_edge(b, edge[2])
        # b.add_edge(a, edge[2])

    # Solve path
    dijkstra(get_node(start), get_node(end))

def get_path_array(node):
    if node.shortest_path_via == None:
        return [node.symbol]
    else:
        return get_path_array(node.shortest_path_via) + [node.symbol]

data = [
    ["A", "B", 1],
    ["A", "C", 3],
    ["B", "C", 1],
    ["C", "A", 1],
]
#solve_dijkstra(data, "A", "C")
#solve_dijkstra(data, "C", "A")


tic = time.perf_counter()

# GIVEN FOR FREE
boxes = np.array([[0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0,],
         [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0,],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,],
         [1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0,],
         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,],
         [1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1,],
         [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,],
         [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0,],
         [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0,],
         [1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0,],
         [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0,]])

walls = np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
         [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
         [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
         [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
         [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
         [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,]])

# IMPORTANT NEW CODE
wx,wy = walls.shape

path_data = []

for x in range(wx):
    for y in range(wy):
        if walls[x][y] != 1:
            if x > 0:
                dist = 1
                if boxes[x-1][y] == 1:
                    dist = 33
                path_data.append([f"{x}-{y}", f"{x-1}-{y}", dist])
            
            if x < wx-1:
                dist = 1
                if boxes[x+1][y] == 1:
                    dist = 33
                path_data.append([f"{x}-{y}", f"{x+1}-{y}", dist])

            if y > 0:
                dist = 1
                if boxes[x][y-1] == 1:
                    dist = 33
                path_data.append([f"{x}-{y}", f"{x}-{y-1}", dist])
            
            if y < wy-1:
                dist = 1
                if boxes[x][y+1] == 1:
                    dist = 33
                path_data.append([f"{x}-{y}", f"{x}-{y+1}", dist])


solve_dijkstra(path_data, "1-1", "9-9")
toc = time.perf_counter()

print(f"{toc-tic:0.4f} seconds")
            
