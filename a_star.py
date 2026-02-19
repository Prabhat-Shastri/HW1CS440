import copy
import heapq
import random
import sys

GRID_SPACE = 20
DEAD_CHANCE = 0.3
TOTAL_CELLS = (GRID_SPACE ** 2)
TOTAL_BOARDS = 50

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


def generateGrid():
    grid = [[-1 for _ in range(GRID_SPACE)] for _ in range(GRID_SPACE)]
    neighbors = [[0,-1],[0,1],[-1,0],[1,0]]
    visited = set()

    numVisited = 0
    while numVisited < TOTAL_CELLS:

        startRow, startCol = random.randint(0, GRID_SPACE - 1), random.randint(0, GRID_SPACE - 1)
        if (startRow, startCol) in visited: continue
        grid[startRow][startCol] = 0

        numVisited += 1

        visited.add((startRow, startCol))
        stack = []
        stack.append((startRow, startCol))

        while stack:
            curRow, curCol = stack.pop()

            if min(curRow, curCol) < 0 or max(curRow, curCol) > (GRID_SPACE - 1):
                continue



            for dr, dc in neighbors:
                newRow, newCol = curRow + dr, curCol + dc

                if min(newRow, newCol) < 0 or max(newRow, newCol) > (GRID_SPACE - 1):
                    continue

                if (newRow, newCol) not in visited:
                    visited.add((newRow, newCol))
                    numVisited += 1

                    if (random.randint(0, 100) / 100) <= 0.3:
                        grid[newRow][newCol] = 1
                        continue
                    grid[newRow][newCol] = 0
                    stack.append((newRow, newCol))
    return grid


# grids = []
# for i in range(TOTAL_BOARDS):
#     grids.append(generateGrid())

# print(f'{TOTAL_BOARDS} grids generated.')


def mDistance(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


grid = generateGrid()
botGrid = [[-1 for _ in range(GRID_SPACE)] for _ in range(GRID_SPACE)]

startRow, startCol = random.randint(0, GRID_SPACE - 1), random.randint(0, GRID_SPACE - 1)
while grid[startRow][startCol] != 0:
    startRow, startCol = random.randint(0, GRID_SPACE - 1), random.randint(0, GRID_SPACE - 1)
start = (startRow,startCol)

goalRow, goalCol = random.randint(0, GRID_SPACE - 1), random.randint(0, GRID_SPACE - 1)
while grid[goalRow][goalCol] != 0 or (goalRow, goalCol) == start:
    goalRow, goalCol = random.randint(0, GRID_SPACE - 1), random.randint(0, GRID_SPACE - 1)
goal = (goalRow, goalCol)

botGrid[startRow][startCol] = 0

minHeap = []
visited = set()
heapq.heappush(minHeap, (0 + mDistance(start, goal), startRow, startCol, 0))

gValues = {
    start: 0
}
parents = {
    start: start
}
neighbors = [[0,-1],[0,1],[-1,0],[1,0]]
print(start)
print(goal)

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

def aStar(start, goal):
    minHeap = []
    visited = set()
    gValues = {start: 0}
    parents = {start: start}
    heapq.heappush(minHeap, (0 + mDistance(start, goal), startRow, startCol, 0))
    while minHeap:
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


cur = start
botGrid[cur[0]][cur[1]] = 0

while True:
    path = aStar(cur, goal)
    print("planned path:", list(reversed(path)))

    if not path:
        print("No path.")
        break


    path_forward = list(reversed(path))

    replanned = False
    for step in path_forward[1:]:
        r, c = step

        if grid[r][c] == 1:
            botGrid[r][c] = 1
            replanned = True
            break

        cur = step
        botGrid[r][c] = 0

        if cur == goal:
            print("Reached goal!")
            replanned = False
            break

    if cur == goal:
        break
    if not replanned:
        break





