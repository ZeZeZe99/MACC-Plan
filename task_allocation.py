
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

def allocate(num, todo, assignment, mode=0):
    if mode == 0:
        return naive_allocation(num, todo, assignment)
    elif mode == 1:
        return dependency_allocation(num, todo, assignment)

def assign_tasks(assignment, tasks, new_id):
    for i in range(len(tasks)):
        assignment[new_id[i]] = tasks[i]
    for i in range(len(assignment)):
        if assignment[i] is None:
            assignment[i] = (-1, -1, -1, -1, -1)
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
    # Allocate tasks to agents
    assignment = assign_tasks(assignment, tasks)
    return assignment, num_dummy

def dependency_allocation(num, g, assignment):
    new_id = [i for i in range(num) if assignment[i] is None]
    leaves = [n for n in g.nodes if g.in_degree[n] == 0]
    candidates = [n for n in leaves if n not in assignment]
    new_tasks = candidates[:min(len(new_id), len(candidates))]
    assignment = assign_tasks(assignment, new_tasks, new_id)
    num_dummy = num - len(new_tasks)
    return assignment, num_dummy
