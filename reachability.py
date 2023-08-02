import numpy as np
import pdb
import heapq
import networkx as nx

class PathFinder:
    def __init__(self, node_map):
        self.map = node_map
        # self.adjacency = self.get_adjacency()
        self.reachability= self.get_reachability(node_map)

    def adjacency_ws(self):
        graph = nx.Graph()
        my_map = self.map
        for i in range(len(my_map)):
            for j in range(len(my_map[0])):
                if i > 0:
                    if abs(my_map[i][j] - my_map[i-1][j]) <= 1:
                        graph.add_edge((i,j), (i-1,j))
                        graph.add_edge((i-1,j), (i,j))
                if i < len(my_map) - 1:
                    if abs(my_map[i][j] - my_map[i+1][j]) <= 1:
                        graph.add_edge((i,j), (i+1,j))
                        graph.add_edge((i+1,j), (i,j))
                if j > 0:
                    if abs(my_map[i][j] - my_map[i][j-1]) <= 1:
                        graph.add_edge((i,j), (i,j-1))
                        graph.add_edge((i,j-1), (i,j))
                if j < len(my_map[0]) - 1:
                    if abs(my_map[i][j] - my_map[i][j+1]) <= 1:
                        graph.add_edge((i,j), (i,j+1))
                        graph.add_edge((i,j+1), (i,j))
        obstacle_map = np.zeros((len(my_map)*len(my_map[0]), len(my_map)*len(my_map[0])))
        for i in range(len(my_map)):
            for j in range(len(my_map[0])):
                if i > 0:
                    if abs(my_map[i][j] - my_map[i-1][j]) <= 1:
                        obstacle_map[i*len(my_map[0])+j, (i-1)*len(my_map[0])+j] = 1
                        obstacle_map[(i-1)*len(my_map[0])+j, i*len(my_map[0])+j] = 1
                if i < len(my_map) - 1:
                    if abs(my_map[i][j] - my_map[i+1][j]) <= 1:
                        obstacle_map[i*len(my_map[0])+j, (i+1)*len(my_map[0])+j] = 1
                        obstacle_map[(i+1)*len(my_map[0])+j, i*len(my_map[0])+j] = 1
                if j > 0:
                    if abs(my_map[i][j] - my_map[i][j-1]) <= 1:
                        obstacle_map[i*len(my_map[0])+j, i*len(my_map[0])+j-1] = 1
                        obstacle_map[i*len(my_map[0])+j-1, i*len(my_map[0])+j] = 1
                if j < len(my_map[0]) - 1:
                    if abs(my_map[i][j] - my_map[i][j+1]) <= 1:
                        obstacle_map[i*len(my_map[0])+j, i*len(my_map[0])+j+1] = 1
                        obstacle_map[i*len(my_map[0])+j+1, i*len(my_map[0])+j] = 1
        return obstacle_map

    def get_reachability(self, node_map, binary_workspace_matrix=None, action_location=[[-1], [-1]]):
        # if action_location is not None:
            # print(action_location[0][0], action_location[1][0])
        # if binary_workspace_matrix is None:
        binary_workspace_matrix = np.pad(np.ones((node_map.shape[0]-2, node_map.shape[1]-2)), ((1, 1), (1, 1)), 'constant', constant_values=0)

        map_loc_nbr_list = dict()
        map_loc_path_list = dict()    
        for i in range(node_map.shape[0]):
            for j in range(node_map.shape[1]):
                map_loc_path_list[(i,j)] = []
                # if binary_workspace_matrix[i,j]==0 and (i,j) != (action_location[0][0], action_location[1][0]):
                if binary_workspace_matrix[i,j]==0:
                    map_loc_path_list[(i,j)].append((-1,-1))
                else:
                    map_loc_nbr_list[(i,j)] = []
                    neighbours = []
                    map_loc_nbr_list[(i,j)].append((i-1,j)) if (i-1 >= 0 and abs(node_map[i,j]-node_map[i-1, j])<=1) else None
                    map_loc_nbr_list[(i,j)].append((i+1,j)) if (i+1 < node_map.shape[0] and abs(node_map[i,j]-node_map[i+1, j])<=1) else None
                    map_loc_nbr_list[(i,j)].append((i,j-1)) if (j-1 >= 0 and abs(node_map[i,j]-node_map[i, j-1])<=1) else None
                    map_loc_nbr_list[(i,j)].append((i,j+1)) if (j+1 < node_map.shape[1] and abs(node_map[i,j]-node_map[i, j+1])<=1) else None
                    for neighbour in map_loc_nbr_list[(i,j)]:
                        if binary_workspace_matrix[neighbour] == 0: #first round of checking if cell is reachable
                            binary_workspace_matrix[i,j] = 0
                            map_loc_path_list[(i,j)].append(neighbour)
    
        flips_count = True
        while(flips_count):
            flips_count = False
            for i in range(node_map.shape[0]):
                for j in range(node_map.shape[1]):
                    if binary_workspace_matrix[i,j]==1: #if cell is unreachable
                        for neighbour in map_loc_nbr_list[(i,j)]: #check if valid neighbours are reachable
                            if binary_workspace_matrix[neighbour]==0: 
                                binary_workspace_matrix[i,j] = 0 #flip if valid neighbours are reachable
                                flips_count = True
                                map_loc_path_list[(i,j)].append(neighbour)
        # else:
        #     # is action location reachable?
        #     x = action_location[0][0]
        #     y = action_location[1][0]
        #     nbr_list = self.nbr_list(x, y)
        #     flips_count = True
           
        #     flip_loc = []
        #     for nbr in nbr_list:
        #         if binary_workspace_matrix[nbr] == 1 and ((node_map[x][y]-node_map[nbr[0], nbr[1]])==0 or (node_map[x][y]-node_map[nbr[0], nbr[1]])==0): # if neighbour was reachable, and height diff <=1
        #             flips_count = False
        #         else:
        #             flip_loc.append((x, y))
            
        #         # is the neighbour itself reachable?
        #         nbr_nbr_list = self.nbr_list(nbr[0], nbr[1])
        #         for nbr_nbr in nbr_nbr_list:
        #             if binary_workspace_matrix[nbr_nbr] == 1 and ((node_map[nbr[0]][nbr[1]]-node_map[nbr_nbr[0], nbr_nbr[1]])==0 or (node_map[nbr[0]][nbr[1]]-node_map[nbr_nbr[0], nbr_nbr[1]])==1):
        #                 flips_count = False
        #             else:
        #                 flip_loc.append((nbr[0], nbr[1]))

        #     while(flips_count):
        #         for loc in list(flip_loc):
        #             nbr_list = self.nbr_list(loc[0], loc[1])
        #             for nbr in nbr_list:
        #                 if binary_workspace_matrix[nbr] == 1 and ((node_map[x][y]-node_map[nbr[0], nbr[1]])==0 or (node_map[x][y]-node_map[nbr[0], nbr[1]])==0):
        #                     flips_count = False
        #                 else:
        #                     flip_loc.append((x, y))
                            

            # did any of the other positions change reachability?

        return binary_workspace_matrix

    # def flip_next(location, binary_workspace_matrix):

    def nbr_list(self, x, y):
        return {(x-1,y), (x+1,y), (x,y-1), (x,y+1)}

    def get_reachability_old(self, node_map, binary_workspace_matrix=None, action_location=None):
        # max_height = 5
        #
        # action_location = argument of non-zero element of self.node_map - node_map
        # action_location = np.nonzero(self.map-node_map)
        self.map = node_map
        # action = node_map[action_location[0], action_location[1]]
        if action_location is not None:
            print(action_location[0][0], action_location[1][0])
        map_loc_nbr_list = dict()
        map_loc_path_list = dict()
        if binary_workspace_matrix is None:    
            binary_workspace_matrix = np.pad(np.ones((node_map.shape[0]-2, node_map.shape[1]-2)), ((1, 1), (1, 1)), 'constant', constant_values=0)
        for i in range(node_map.shape[0]):
            for j in range(node_map.shape[1]):
                map_loc_path_list[(i,j)] = []
                if binary_workspace_matrix[i,j]==0:
                    map_loc_path_list[(i,j)].append((-1,-1))
                if binary_workspace_matrix[i,j]==1:
                    map_loc_nbr_list[(i,j)] = []
                    neighbours = []
                    map_loc_nbr_list[(i,j)].append((i-1,j)) if (i-1 >= 0 and abs(node_map[i,j]-node_map[i-1, j])<=1) else None
                    map_loc_nbr_list[(i,j)].append((i+1,j)) if (i+1 < node_map.shape[0] and abs(node_map[i,j]-node_map[i+1, j])<=1) else None
                    map_loc_nbr_list[(i,j)].append((i,j-1)) if (j-1 >= 0 and abs(node_map[i,j]-node_map[i, j-1])<=1) else None
                    map_loc_nbr_list[(i,j)].append((i,j+1)) if (j+1 < node_map.shape[1] and abs(node_map[i,j]-node_map[i, j+1])<=1) else None
                    for neighbour in map_loc_nbr_list[(i,j)]:
                        if binary_workspace_matrix[neighbour] == 0: #first round of checking if cell is reachable
                            binary_workspace_matrix[i,j] = 0
                            map_loc_path_list[(i,j)].append(neighbour)
        
        flips_count = True
        while(flips_count):
            flips_count = False
            for i in range(node_map.shape[0]):
                for j in range(node_map.shape[1]):
                    if binary_workspace_matrix[i,j]==1: #if cell is unreachable
                        for neighbour in map_loc_nbr_list[(i,j)]: #check if valid neighbours are reachable
                            if binary_workspace_matrix[neighbour]==0: 
                                binary_workspace_matrix[i,j] = 0 #flip if valid neighbours are reachable
                                flips_count = True
                                map_loc_path_list[(i,j)].append(neighbour)

        # print("map_loc_path_list ", map_loc_path_list)
        return binary_workspace_matrix

    # def main():
    #     node_map = np.zeros((5,5))
    #     node_map = [[0,0,0,0,0],
    #                 [0,0,0,0,0],
    #                 [0,1,2,2,0],
    #                 [0,0,0,0,0],
    #                 [0,0,0,0,0]]
    #     node_map[1,3] = 1
    #     # get_reachability(node_map)
    #     # print("reachability ", get_reachability(node_map))

    # if __name__ == "__main__":
    #     main()