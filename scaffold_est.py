from copy import deepcopy
import numpy as np

import lego
import config

def find_groups(env, height):
    """
    Find groups of unfinished goals, by goal level
    Groups: a numpy array of shape (H-1, w, w)
    """
    groups = []
    for lv in range(1, env.H):
        workspace = env.goal3d[lv] * (height <= lv)  # Unfinished goals at level lv
        group_map = np.zeros((env.w, env.w), dtype=np.int8)
        gid = 1
        for x in range(1, env.w - 1):
            for y in range(1, env.w - 1):
                if workspace[x, y] > 0:
                    g_map = np.zeros((env.w, env.w), dtype=np.int32)
                    connect_goals(env, workspace, x, y, g_map, val=1)
                    group_map += g_map * gid
                    gid += 1
        groups.append(group_map)
    groups = np.stack(groups, axis=0)
    return groups

def connect_goals(env, workspace, x, y, g_map, val=1, new_val=0):
    """Connect neighboring goals into a group"""
    if workspace[x, y] != val:
        return
    workspace[x, y] = new_val
    g_map[x, y] = 1
    for (x2, y2) in env.search_neighbor[(x, y)]:
        connect_goals(env, workspace, x2, y2, g_map, val, new_val)

def find_group_support(env, groups):
    """
    Find the d-support set of each group. Also count size of each group.
    For each group, each d-support set is split into two parts:
        1) red region: supports with minimum horizontal distance to group < d
        2) blue region: supports with minimum horizontal distance to group = d
    Return a list of support maps, each map is of shape (num_groups, H - 1, 2, w, w)
    Return a list of group sizes, each of shape (num_groups,)
    """
    group2support = []
    sizes = []
    for lv in range(1, env.H):  # For goals at each level
        group_map = groups[lv - 1]
        gids = np.unique(group_map)[1:]
        num = len(gids)
        size = np.zeros((num), dtype=np.int8)
        support_map = np.zeros((num, env.H - 1, 2, env.w, env.w), dtype=np.int8)
        for i, g in enumerate(gids):  # For each group
            g_map = group_map == g
            s_map = support_map[i]
            for slv in range(lv):
                for (x, y) in np.transpose(np.nonzero(group_map == g)):
                    s_map[slv, 0] |= env.goal2support[(lv, x, y)][slv]
                s_map[slv, 0] *= (1 - g_map)
            for slv in range(lv):  # Split the support set into red and blue regions
                if slv + 2 < lv:
                    s_map[slv, 1] = s_map[slv, 0] * (1 - s_map[slv + 2, 0]) * (1 - s_map[slv + 1, 0])
                elif slv + 1 < lv:
                    s_map[slv, 1] = s_map[slv, 0] * (1 - s_map[slv + 1, 0])
                else:
                    s_map[slv, 1] = s_map[slv, 0]
                s_map[slv, 0] -= s_map[slv, 1]
            size[i] = np.count_nonzero(g_map)
        group2support.append(support_map)
        sizes.append(size)
    return group2support, sizes

def cast_scaffold_value(env, info):
    """
    For goal groups at each level, cast scaffold value to levels below, and record useful d-support
    scaffold_v: a numpy array of shape (H - 1, H - 1, w, w)
    useful_support: a list of useful entry, each of shape (num_groups, H - 1)
    """
    scaffold_val = np.zeros((env.H - 1, env.H - 1, env.w, env.w), dtype=np.int8)
    useful_support = []
    for lv in range(1, env.H):  # For goals at each level
        val = scaffold_val[lv - 1]
        supports = info['group2support'][lv - 1]
        sizes = info['sizes'][lv - 1]
        num = supports.shape[0]
        useful = np.zeros((num, env.H - 1), dtype=np.int8)
        for i in range(num):  # For each group
            for slv in range(lv):
                lv_support = supports[i, slv]
                _, u, v = cal_scaffold_val(slv, lv_support, info['world'], env.goal3d, lv, sizes[i])
                val[slv] += v
                useful[i, slv] = u
        useful_support.append(useful)
    return scaffold_val, useful_support

def cal_scaffold_val(lv, support, world, goal3d, glv, g_num):
    """
    Calculate the scaffold value
        Red region: 0
        Blue region: 1 if not enough useful supports exist; 0 otherwise
    A support cell within the support set is useful if:
        1. It's an added scaffold block, or
        2. It's an un-added goal block, or
        3. It's an added goal block with no goal block added above
    Threshold for useful supports:
        1) 1 in blue region, or
        2) n in red region, n = goal lv - support lv - 1 * (# goal = 1)
    """
    block = world[0, lv]
    goal_above = world[1, lv + 1]
    goal = goal3d[lv]
    useful_blue = support[1] * (block | goal) * (1 - goal_above)
    if useful_blue.sum() > 0:
        return True, 0, 0
    useful_red = support[0] * (block | goal) * (1 - goal_above)
    if useful_red.sum() >= max(1, glv - lv - (g_num == 1)):
        return True, 0, 0
    valid = ((support[0] | support[1]) & (1 - goal_above)).any()
    return valid, 1, support[1] * (1 - goal_above)


'''Incremental goal value update'''
def init_scaffold_info(env, height):
    info = dict()
    info['groups'] = find_groups(env, height)
    info['group2support'], info['sizes'] = find_group_support(env, info['groups'])
    info['world'] = np.zeros((2, env.H, env.w, env.w), dtype=np.int8)
    info['scaffold_val'], info['useful_support'] = cast_scaffold_value(env, info)
    return info

def update_scaffold_info(env, info, add, loc, z, world):
    info = deepcopy(info)
    info['world'] = world
    is_goal = env.goal3d[z, loc[0], loc[1]] == 1
    high_goal = is_goal and z > 0
    if high_goal:
        update_group(env, info, add, loc, z)
        update_group_support(env, info, z)
    valid = update_scaffold_value(env, info, z, high_goal)
    return valid, info

def update_group(env, info, add, loc, z):
    """Update groups of unfinished goals"""
    group_map = info['groups'][z - 1]
    if add:  # Added a goal block, check if it breaks a group into two
        gid = group_map[loc]
        assert gid != 0
        new_id = group_map.max() + 1
        group_map[loc] = 0
        for nx, ny in env.search_neighbor[loc]:
            g_map = np.zeros((env.w, env.w), dtype=np.int32)
            if group_map[nx, ny] == gid:
                connect_goals(env, group_map, nx, ny, g_map, val=gid)
                if np.any(group_map == gid):  # Has another part with value gid
                    group_map += g_map * new_id
                    new_id += 1
                else:
                    group_map += g_map * gid
                    break
    else:  # Removed a goal block, check if it merges two groups into one
        gid = 0
        for nbr in env.search_neighbor[loc]:
            nid = group_map[nbr]
            if nid > 0:
                if gid == 0:
                    gid = nid
                elif gid != nid:
                    group_map = np.where(group_map == nid, gid, group_map)
        if gid != 0:
            group_map[loc] = gid
        else:
            group_map[loc] = group_map.max() + 1
    info['groups'][z - 1] = group_map

def update_group_support(env, info, z):
    group_map = info['groups'][z - 1]
    gids = np.unique(group_map)[1:]
    num = len(gids)
    new_size = np.zeros(num, dtype=np.int8)
    support_map = np.zeros((num, env.H - 1, 2, env.w, env.w), dtype=np.int8)
    for i, g in enumerate(gids):  # For each group
        g_map = group_map == g
        s_map = support_map[i]
        for slv in range(z):
            for (x, y) in np.transpose(np.nonzero(group_map == g)):
                s_map[slv, 0] |= env.goal2support[(z, x, y)][slv]
            s_map[slv, 0] *= (1 - g_map)
        for slv in range(z):  # Split the support set into red and blue regions
            if slv + 2 < z:
                s_map[slv, 1] = s_map[slv, 0] * (1 - s_map[slv + 2, 0]) * (1 - s_map[slv + 1, 0])
            elif slv + 1 < z:
                s_map[slv, 1] = s_map[slv, 0] * (1 - s_map[slv + 1, 0])
            else:
                s_map[slv, 1] = s_map[slv, 0]
            s_map[slv, 0] -= s_map[slv, 1]
        new_size[i] = np.count_nonzero(g_map)
    info['group2support'][z - 1] = support_map
    info['sizes'][z - 1] = new_size

def update_scaffold_value(env, info, z, is_goal):
    """
    Update scaffold value and useful support when an action is taken at level z
    1. Goals above level z may cast different scaffold value to level z
    2. If the action is a goal:
        2.1. Goals above level z may cast different scaffold value to level z - 1
        2.2. Goals at level z may cast different scaffold value to levels below z
    """
    for lv in range(z + 1, env.H):  # Goals above level z
        zs = [z, z - 1] if is_goal and z > 0 else [z]
        for slv in zs:
            val = np.zeros(env.world_shape, dtype=np.int8)
            useful = info['useful_support'][lv - 1]
            support = info['group2support'][lv - 1]
            size = info['sizes'][lv - 1]
            num = useful.shape[0]
            for i in range(num):  # For each group
                lv_support = support[i, slv]
                valid, u, v = cal_scaffold_val(slv, lv_support, info['world'], env.goal3d, lv, size[i])
                if not valid:
                    return False
                val += v
                useful[i, slv] = u
            info['scaffold_val'][lv - 1, slv] = val
    if is_goal:
        val = np.zeros((env.H - 1, env.w, env.w), dtype=np.int8)
        support = info['group2support'][z - 1]
        size = info['sizes'][z - 1]
        num = support.shape[0]
        useful = np.zeros((num, env.H - 1), dtype=np.int8)
        for i in range(num):  # For each group
            for slv in range(z):
                lv_support = support[i, slv]
                valid, u, v = cal_scaffold_val(slv, lv_support, info['world'], env.goal3d, z, size[i])
                if not valid:
                    return False
                val[slv] += v
                useful[i, slv] = u
        info['scaffold_val'][z - 1] = val
        info['useful_support'][z - 1] = useful
    return True

def cal_goal_val(env, info):
    group2support = info['group2support']
    useful_support = info['useful_support']
    scaffold_v = info['scaffold_val'].sum(axis=0)
    goal_v = 0
    for lv in range(1, env.H):  # For goals at each level
        support_lv = group2support[lv - 1].sum(axis=2)
        useful_lv = useful_support[lv - 1]
        num = useful_lv.shape[0]
        for i in range(num):  # For each group
            val = (scaffold_v * support_lv[i]).max(axis=(1, 2))
            val = np.divide(useful_lv[i], val, out=np.zeros_like(val, dtype=np.float32), where=val > 0)
            goal_v += val.sum()
    return goal_v


if __name__ == '__main__':
    arg = config.get_parser()
    arg = arg.parse_args()

    lego_env = lego.GridWorld(arg)
    lego_env.set_goal()
    lego_env.set_shadow()
    lego_env.set_distance_map()
    lego_env.set_support_map()
