
"""
Task selection methods
Method 1: Naive selection
    1) Select the first N tasks from the high-level plan
    2) Remove tasks that deal with the same block as another task
    3) Fill up the remaining tasks with dummy tasks
Method 2: Dependency selection (1 level in a round)
    1) Select the first N tasks from the current lowest level of the dependency graph
    2) Fill up the remaining tasks with dummy tasks

"""

def allocate(num, todo, assignment):
    return naive_allocation(num, todo, assignment)

def assign_tasks(num, assignment, tasks):
    i = 0
    for a in range(num):
        if assignment[a] is None:
            while tasks[i] in assignment:
                i += 1
            assignment[a] = tasks[i]
    return assignment

def naive_allocation(num, todo, assignment):
    # Select candidate tasks
    candidate = todo[:min(num, len(todo))]
    # Delay a task if it deals with the same block as another task
    tasks = []
    for i in range(len(candidate)):
        same_block = False
        for j in range(i - 1):
            if candidate[i][1:] == candidate[j][1:]:
                same_block = True
                break
        if not same_block:
            tasks.append(candidate[i])
    # Append dummy tasks if there are fewer tasks than agents
    num_dummy = num - len(tasks)
    tasks += num_dummy * [(-1, -1, -1, -1)]
    # Allocate tasks to agents
    assignment = assign_tasks(num, assignment, tasks)
    return assignment, num_dummy

def dependency_allocation(num, g, assignment):
    leaves = [n for n in g.nodes if g.out_degree(n) == 0]
    if len(leaves) < num:
        num_dummy = num - len(leaves)
        tasks = leaves + num_dummy * [(-1, -1, -1, -1)]
    else:
        num_dummy = 0
        tasks = leaves[:num]

