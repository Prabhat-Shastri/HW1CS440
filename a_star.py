import copy
import heapq
import random
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

GRID_SPACE = 101
DEAD_CHANCE = 0.3
TOTAL_CELLS = (GRID_SPACE ** 2)
TOTAL_BOARDS = 50
GRIDWORLDS_DIR = "gridworlds"

def show_grid_terminal(grid, start=None, goal=None, unknown=-1, safe=0, blocked=1):
    BG_RED   = "\x1b[48;2;255;0;0m"
    BG_WHITE = "\x1b[48;2;255;255;255m"
    BG_BLACK = "\x1b[48;2;0;0;0m"
    BG_GREEN = "\x1b[48;2;0;255;0m"
    BG_MAG   = "\x1b[48;2;255;0;255m"
    RESET    = "\x1b[0m"

    start = tuple(start) if start is not None else None
    goal  = tuple(goal)  if goal  is not None else None

    out = []
    for r, row in enumerate(grid):
        line = []
        for c, v in enumerate(row):
            if start is not None and (r, c) == start:
                line.append(BG_GREEN + "  ")
            elif goal is not None and (r, c) == goal:
                line.append(BG_GREEN + "  ")
            elif v == unknown:
                line.append(BG_RED + "  ")
            elif v == safe:
                line.append(BG_WHITE + "  ")
            elif v == blocked:
                line.append(BG_BLACK + "  ")
            else:
                line.append(BG_MAG + "  ")
        line.append(RESET)
        out.append("".join(line))

    sys.stdout.write("\n".join(out) + RESET + "\n\n")


def grid_to_rgb(grid, unknown=-1, safe=0, blocked=1):
    """Convert grid (list of lists: unknown/safe/blocked) to RGB array for imshow."""
    rows, cols = len(grid), len(grid[0])
    out = np.zeros((rows, cols, 3))
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c] if isinstance(grid[r][c], (int, float)) else grid[r][c]
            if v == unknown:
                out[r, c] = [0.85, 0.3, 0.3]   # red — unobserved
            elif v == safe:
                out[r, c] = [1.0, 1.0, 1.0]    # white — unblocked
            else:
                out[r, c] = [0.15, 0.15, 0.15] # black — blocked
    return out


def visualize_demo_matplotlib():
    """Run Repeated Forward A* on gridworld 1 with matplotlib animation: imshow + plot(path) + scatter(start, goal, agent)."""
    ensure_gridworlds_saved()
    path1 = os.path.join(GRIDWORLDS_DIR, "gridworld_1.txt")
    if not os.path.isfile(path1):
        print("(Matplotlib demo skipped: gridworld_1.txt not found.)")
        return

    grid = loadGrid(path1)
    start, goal = generateStates(grid)
    botGrid = [[-1 for _ in range(GRID_SPACE)] for _ in range(GRID_SPACE)]
    cur = start
    totalExpanded = 0

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plt.ion()
    ax.set_title("Repeated Forward A* — agent view (red=unobserved, white=unblocked, black=blocked)")
    ax.set_xlabel("col")
    ax.set_ylabel("row")

    def draw(cur_pos, path_forward, bot, interval=0.02):
        arr = grid_to_rgb(bot)
        ax.clear()
        # Explicit extent so plot() and scatter() use the same coordinates as the image
        extent = (0, GRID_SPACE, GRID_SPACE, 0)  # (left, right, bottom, top) for origin='upper'
        ax.imshow(arr, origin="upper", interpolation="nearest", aspect="equal", extent=extent)
        ax.set_title("Repeated Forward A* — agent view")
        if path_forward:
            # Plot through cell centers so the path aligns with the grid and is fully visible
            path_x = [p[1] + 0.5 for p in path_forward]
            path_y = [p[0] + 0.5 for p in path_forward]
            ax.plot(path_x, path_y, "b-", linewidth=3, label="path", zorder=4)
        # Scatter at cell centers
        ax.scatter([start[1] + 0.5], [start[0] + 0.5], c="lime", s=120, marker="s", edgecolors="darkgreen", linewidths=2, label="start", zorder=5)
        ax.scatter([goal[1] + 0.5], [goal[0] + 0.5], c="cyan", s=120, marker="*", edgecolors="darkblue", linewidths=1, label="goal", zorder=5)
        ax.scatter([cur_pos[1] + 0.5], [cur_pos[0] + 0.5], c="orange", s=100, marker="o", edgecolors="red", linewidths=2, label="agent", zorder=5)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlim(0, GRID_SPACE)
        ax.set_ylim(GRID_SPACE, 0)
        ax.set_aspect("equal")
        plt.pause(interval)

    while cur != goal:
        path, expanded = aStarTieBreaker(botGrid, cur, goal, "l")
        totalExpanded += expanded

        if not path:
            print("No path.")
            draw(cur, [], botGrid, interval=0.5)
            plt.ioff()
            plt.show()
            return

        path_forward = list(reversed(path))

        for step in path_forward[1:]:
            observe_neighbors(grid, botGrid, cur[0], cur[1])
            draw(cur, path_forward, botGrid)

            r, c = step
            if grid[r][c] == 1:
                botGrid[r][c] = 1
                draw(cur, path_forward, botGrid, interval=0.3)
                break
            cur = step
            botGrid[r][c] = grid[r][c]
            if cur == goal:
                draw(cur, path_forward, botGrid, interval=0.5)
                print("Reached goal!")
                print(f"Total expansions: {totalExpanded}")
                plt.ioff()
                plt.show()
                return

    plt.ioff()
    plt.show()
    print(f"Total expansions: {totalExpanded}")


def generateGrid():
    # Part 0: maze/corridor-like with DFS, random tie breaking, 30% blocked / 70% unblocked.
    grid = [[-1 for _ in range(GRID_SPACE)] for _ in range(GRID_SPACE)]
    neighbors = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    visited = set()

    def getUnvisited():
        pool = [(r, c) for r in range(GRID_SPACE) for c in range(GRID_SPACE) if (r, c) not in visited]
        return random.choice(pool) if pool else None

    def getNeighbors(r, c):
        out = []
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < GRID_SPACE and 0 <= nc < GRID_SPACE and (nr, nc) not in visited:
                out.append((nr, nc))
        return out

    while len(visited) < TOTAL_CELLS:
        if not visited:
            startRow = random.randint(0, GRID_SPACE - 1)
            startCol = random.randint(0, GRID_SPACE - 1)
        else:
            startRow, startCol = getUnvisited()

        visited.add((startRow, startCol))
        grid[startRow][startCol] = 0  # seed always unblocked
        stack = [(startRow, startCol)]

        while stack:
            curRow, curCol = stack[-1]  # peek — backtrack only when dead-end
            nbrs = getNeighbors(curRow, curCol)

            if not nbrs:
                stack.pop()
                continue

            newRow, newCol = random.choice(nbrs)  # random tie breaking
            visited.add((newRow, newCol))

            if random.random() < DEAD_CHANCE:
                grid[newRow][newCol] = 1  # blocked — do not push
            else:
                grid[newRow][newCol] = 0
                stack.append((newRow, newCol))

    return grid


def saveGrid(grid, path):
    with open(path, 'w') as f:
        for row in grid:
            f.write(' '.join(str(cell) for cell in row) + '\n')


def loadGrid(path):
    with open(path) as f:
        return [[int(x) for x in line.split()] for line in f]


# grids = []
# for i in range(TOTAL_BOARDS):
#     grids.append(generateGrid())

# print(f'{TOTAL_BOARDS} grids generated.')

def generateStates(grid):
    startRow, startCol = random.randint(0, GRID_SPACE - 1), random.randint(0, GRID_SPACE - 1)
    while grid[startRow][startCol] != 0:
        startRow, startCol = random.randint(0, GRID_SPACE - 1), random.randint(0, GRID_SPACE - 1)
    start = (startRow,startCol)

    goalRow, goalCol = random.randint(0, GRID_SPACE - 1), random.randint(0, GRID_SPACE - 1)
    while grid[goalRow][goalCol] != 0 or (goalRow, goalCol) == start:
        goalRow, goalCol = random.randint(0, GRID_SPACE - 1), random.randint(0, GRID_SPACE - 1)
    goal = (goalRow, goalCol)

    return start, goal

def mDistance(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

NEIGHBORS_4 = [(0, -1), (0, 1), (-1, 0), (1, 0)]

def observe_neighbors(grid, botGrid, r, c):
    """Agent observes all 4 adjacent cells at (r,c) and updates botGrid from grid."""
    for dr, dc in NEIGHBORS_4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < GRID_SPACE and 0 <= nc < GRID_SPACE:
            botGrid[nr][nc] = grid[nr][nc]

def constructPath(path, goal):
    if goal not in path:
        return []
    
    p = []
    cur = goal
    p.append(goal)

    while cur in path and path[cur] != cur:
        cur = path[cur]
        p.append(cur)
    return p

def aStar(botGrid, s, goal):
    neighbors = [[0,-1],[0,1],[-1,0],[1,0]]

    minHeap = []
    visited = set()
    gValues = {s: 0}
    parents = {s: s}
    heapq.heappush(minHeap, (0 + mDistance(s, goal), s[0], s[1], 0))
    i = 0
    while minHeap:
        i+=1
        curCost, curRow, curCol, curG= heapq.heappop(minHeap)

        if curG != gValues[(curRow, curCol)]: continue

        if (curRow, curCol) == goal:
            return constructPath(parents, goal)

        
        
        visited.add((curRow, curCol))

        for dr, dc in neighbors:
            newRow, newCol = curRow + dr, curCol + dc

            if min(newRow, newCol) < 0 or max(newRow, newCol) > (GRID_SPACE - 1):
                continue

            if ((newRow, newCol)) in visited: 
                continue

            if botGrid[newRow][newCol] == 1:
                continue

            newG = gValues[(curRow, curCol)] + 1

            if newG < gValues.get((newRow, newCol), float('inf')):
                heapq.heappush(minHeap, (newG + mDistance((newRow, newCol), goal), newRow, newCol, newG))
                gValues[(newRow, newCol)] = newG
                parents[(newRow, newCol)] = (curRow, curCol)
    return constructPath(parents, goal)

def aStarTieBreaker(botGrid, s, goal, tieType: str):
    neighbors = [[0,-1],[0,1],[-1,0],[1,0]]

    C = (GRID_SPACE ** 2)

    def tieBreaker(tieType, st, gl, gVal):
        d = mDistance(st, gl)
        # 'l' = larger g first (priority C*f - g); 's' = smaller g first (priority C*f + g)
        return (C * (gVal + d)) - gVal if tieType == 'l' else (C * (gVal + d)) +gVal

    minHeap = []
    visited = set()
    gValues = {s: 0}
    parents = {s: s}
    heapq.heappush(minHeap, (tieBreaker(tieType=tieType, st=s, gl=goal, gVal=0), s[0], s[1], 0))
    i = 0
    while minHeap:
        i+=1
        curCost, curRow, curCol, curG= heapq.heappop(minHeap)

        if curG != gValues[(curRow, curCol)]: continue

        if (curRow, curCol) == goal:
            return constructPath(parents, goal), len(visited)

        
        
        visited.add((curRow, curCol))

        for dr, dc in neighbors:
            newRow, newCol = curRow + dr, curCol + dc

            if min(newRow, newCol) < 0 or max(newRow, newCol) > (GRID_SPACE - 1):
                continue

            if ((newRow, newCol)) in visited: 
                continue

            if botGrid[newRow][newCol] == 1:
                continue

            newG = gValues[(curRow, curCol)] + 1

            if newG < gValues.get((newRow, newCol), float('inf')):
                heapq.heappush(minHeap, (tieBreaker(tieType=tieType, st=(newRow, newCol), gl=goal, gVal=newG), newRow, newCol, newG))
                gValues[(newRow, newCol)] = newG
                parents[(newRow, newCol)] = (curRow, curCol)
    return constructPath(parents, goal), len(visited)


def aStarBackwardTieBreaker(botGrid, agentCur, goal, tieType: str):
    # Part 3: Backward A* — search from goal (target) toward agentCur (agent). Path returned is agentCur -> goal.
    neighbors = [[0,-1],[0,1],[-1,0],[1,0]]
    C = (GRID_SPACE ** 2)
    searchStart = goal
    searchGoal = agentCur

    def tieBreaker(tieType, st, gl, gVal):
        d = mDistance(st, gl)
        return (C * (gVal + d)) - gVal if tieType == 'l' else (C * (gVal + d)) + gVal

    minHeap = []
    visited = set()
    gValues = {searchStart: 0}
    parents = {searchStart: searchStart}
    heapq.heappush(minHeap, (tieBreaker(tieType=tieType, st=searchStart, gl=searchGoal, gVal=0), searchStart[0], searchStart[1], 0))

    while minHeap:
        curCost, curRow, curCol, curG = heapq.heappop(minHeap)

        if curG != gValues[(curRow, curCol)]:
            continue

        if (curRow, curCol) == searchGoal:
            path = constructPath(parents, searchGoal)
            return path, len(visited)

        visited.add((curRow, curCol))

        for dr, dc in neighbors:
            newRow, newCol = curRow + dr, curCol + dc

            if min(newRow, newCol) < 0 or max(newRow, newCol) > (GRID_SPACE - 1):
                continue

            if (newRow, newCol) in visited:
                continue

            if botGrid[newRow][newCol] == 1:
                continue

            newG = gValues[(curRow, curCol)] + 1

            if newG < gValues.get((newRow, newCol), float('inf')):
                heapq.heappush(minHeap, (tieBreaker(tieType=tieType, st=(newRow, newCol), gl=searchGoal, gVal=newG), newRow, newCol, newG))
                gValues[(newRow, newCol)] = newG
                parents[(newRow, newCol)] = (curRow, curCol)

    return constructPath(parents, searchGoal), len(visited)





def repeatedAStar(t: str, grid=None, start=None, goal=None, return_bot_grid=False):
    if grid is None:
        grid = generateGrid()
    if start is None or goal is None:
        start, goal = generateStates(grid)
    botGrid = [[-1 for _ in range(GRID_SPACE)] for _ in range(GRID_SPACE)]

    cur = start
    totalExpanded = 0

    while cur != goal:
        path, expanded = aStarTieBreaker(botGrid, cur, goal, t)
        totalExpanded += expanded

        if not path:
            print("No path.")
            return (totalExpanded, botGrid) if return_bot_grid else totalExpanded

        path_forward = list(reversed(path))

        for step in path_forward[1:]:
            observe_neighbors(grid, botGrid, cur[0], cur[1])
            r, c = step
            if grid[r][c] == 1:
                botGrid[r][c] = 1
                break
            cur = step
            botGrid[r][c] = grid[r][c]
            if cur == goal:
                print("Reached goal!")
                return (totalExpanded, botGrid) if return_bot_grid else totalExpanded

    return (totalExpanded, botGrid) if return_bot_grid else totalExpanded


def repeatedBackwardAStar(t: str, grid=None, start=None, goal=None):
    if grid is None:
        grid = generateGrid()
    if start is None or goal is None:
        start, goal = generateStates(grid)
    botGrid = [[-1 for _ in range(GRID_SPACE)] for _ in range(GRID_SPACE)]

    cur = start
    totalExpanded = 0

    while cur != goal:
        path, expanded = aStarBackwardTieBreaker(botGrid, cur, goal, t)
        totalExpanded += expanded

        if not path:
            print("No path.")
            return totalExpanded

        # Backward search: constructPath(parents, searchGoal) walks from agentCur toward goal,
        # so path is already [agentCur, ..., goal]. No reversal needed.
        path_forward = path

        for step in path_forward[1:]:
            observe_neighbors(grid, botGrid, cur[0], cur[1])
            r, c = step
            if grid[r][c] == 1:
                botGrid[r][c] = 1
                break
            cur = step
            botGrid[r][c] = grid[r][c]
            if cur == goal:
                print("Reached goal!")
                return totalExpanded

    return totalExpanded


def ensure_gridworlds_saved():
    """Generate and save 50 gridworlds if not present; return paths to the 50 files."""
    os.makedirs(GRIDWORLDS_DIR, exist_ok=True)
    paths = [os.path.join(GRIDWORLDS_DIR, f"gridworld_{i}.txt") for i in range(1, TOTAL_BOARDS + 1)]
    if all(os.path.isfile(p) for p in paths):
        return paths
    for i, path in enumerate(paths):
        g = generateGrid()
        saveGrid(g, path)
        print(f"Saved {path}")
    return paths


def FiftyBoardRepeatedAStarTieBreaker():
    paths = ensure_gridworlds_saved()
    grids = [loadGrid(p) for p in paths]
    # t='l' = break ties in favor of larger g; t='s' = smaller g. Same start/goal per grid for fair comparison.
    larger_g_tie, smaller_g_tie = 0, 0
    for grid in grids:
        start, goal = generateStates(grid)
        larger_g_tie += repeatedAStar(t='l', grid=grid, start=start, goal=goal)
        smaller_g_tie += repeatedAStar(t='s', grid=grid, start=start, goal=goal)
    return (larger_g_tie / TOTAL_BOARDS, smaller_g_tie / TOTAL_BOARDS)


def FiftyBoardForwardVsBackward():
    paths = ensure_gridworlds_saved()
    grids = [loadGrid(p) for p in paths]
    forward_exp, backward_exp = 0, 0
    for grid in grids:
        start, goal = generateStates(grid)
        forward_exp += repeatedAStar(t='l', grid=grid, start=start, goal=goal)
        backward_exp += repeatedBackwardAStar(t='l', grid=grid, start=start, goal=goal)
    return (forward_exp / TOTAL_BOARDS, backward_exp / TOTAL_BOARDS)


out = FiftyBoardRepeatedAStarTieBreaker()
print(out)

part3 = FiftyBoardForwardVsBackward()
print("Part 3 — Forward vs Backward (same 50 grids, tie-break larger g):")
print("  Repeated Forward A*:  avg expansions =", part3[0])
print("  Repeated Backward A*:  avg expansions =", part3[1])

# ─── Visualization demo: matplotlib (imshow + plot path + scatter start/goal/agent) ───
RUN_VISUALIZATION_DEMO = True
if RUN_VISUALIZATION_DEMO:
    visualize_demo_matplotlib()
