import copy
import heapq
import random
import sys

GRID_SPACE = 101
DEAD_CHANCE = 0.3
TOTAL_CELLS = (GRID_SPACE ** 2)
TOTAL_BOARDS = 50


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



def aStarTieBreaker(botGrid, s, goal, tieType: str = 'l'):
    def mDistance(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

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
    neighbors = [[0,-1],[0,1],[-1,0],[1,0]]

    C = (GRID_SPACE ** 2)

    def tieBreaker(tieType, st, gl, gVal):
        d = mDistance(st, gl)
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





def repeatedAStar(t: str, forward: bool = True):
    grid = generateGrid()
    botGrid = [[-1 for _ in range(GRID_SPACE)] for _ in range(GRID_SPACE)]

    start, goal = generateStates(grid)
    cur = start

    totalExpanded = 0

    while True:
        path, expanded = aStarTieBreaker(botGrid, cur, goal, t)

        totalExpanded += expanded

        if not path:
            print("No path.")
            return totalExpanded


        path_backward = copy.deepcopy(path)
        path_forward = list(reversed(path))
        
        truePath = path_forward if forward else path_backward

        replanned = False
        for step in truePath[1:]:
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
                return totalExpanded

        if cur == goal:
            break
        if not replanned:
            break
    return


def FiftyBoardRepeatedAStarTieBreaker():
    big_g_tie, little_g_tie = 0, 0
    for i in range(TOTAL_BOARDS):
        big_g_tie += repeatedAStar(t='l')
        little_g_tie += repeatedAStar(t='s')

    return (big_g_tie / TOTAL_BOARDS, little_g_tie / TOTAL_BOARDS)


out = FiftyBoardRepeatedAStarTieBreaker()
print(out)





