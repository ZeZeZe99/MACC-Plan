import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import dep_graph as dg
from mpl_toolkits.mplot3d import Axes3D
import pdb

# animate only the high level steps for vanilla experiment
def high_level_plan_vanilla():
    graph = dg.main()
    graph.adg.remove_node((-1, -1, -1, 0))
    # skip data[0] as empty node
    i = 1
    axes = [len(graph.adg.nodes())+1, 8, 8]
    data = np.zeros(axes)
    while(len(graph.adg.nodes())>0):
        node_list = []
        for node in graph.adg.nodes():
            if len(graph.adg.pred[node]) == 0:
                data[i] = data[i-1]
                data[i][node[0]][node[1]]+=node[3]
                i+=1
                node_list.append(node)
        for node in node_list:
            graph.adg.remove_node(node)
    print(data)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for i in range(1, len(data)):
        ax.cla()
        ax.set_title('Step '+str(i))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 8)
        ax.set_zlim(0, 8)
        matrix = np.zeros((8, 8, 8), dtype=bool)
        for x in range(8):
            for y in range(8):
                for z in range(8):
                    if data[i][x][y] > 0 and z < data[i][x][y]:
                        matrix[x][y][z] = True
        ax.voxels(matrix, edgecolor='k')
        plt.pause(1)
    plt.show()

        
if __name__ == '__main__':
    high_level_plan_vanilla()
# animate only the high level steps for space experiments

# animate low-level for vanilla

# animate low-level for space
