import time
from typing import List, Tuple
import gymnasium as gym

MOVE_DOWN, MOVE_UP, MOVE_RIGHT, MOVE_LEFT, PICKUP, DROPOFF = range(6)
DICT_ACTIONS = {0: "⬇️", 1: "⬆️", 2: "➡️", 3: "⬅️", 4: "👩", 5: "🏢"}
DICT_REWARDS = {-1: "❌", -10: "❌", 20: "✅"}
CYAN = "\033[96m"
OFF = "\033[0m"


class Bruteforce:
    def __init__(self) -> None:
        self.env: gym.Env = gym.make("Taxi-v3", render_mode="human")
        self.steps: int = 0
        self.obstacles = set()
        self.visited = set()
        self.clientDropoff: bool = False
        self.rewards: int = 0
        self.invertDirections = {
            MOVE_DOWN: MOVE_UP,
            MOVE_UP: MOVE_DOWN,
            MOVE_RIGHT: MOVE_LEFT,
            MOVE_LEFT: MOVE_RIGHT,
        }
        self.previous_positions: List[int] = []
        self.fromHsitory: bool = False

        state = self.env.reset()[0]

        self.rows, self.cols = self.env.unwrapped.desc.shape
        self.rows -= 2
        self.cols //= 2
        self.totalMapPositions = self.rows * self.cols

        taxi_row, taxi_col, _, _ = self.env.unwrapped.decode(state)
        self.taxiPos = (taxi_row, taxi_col)

    def drawMap(self, playerX: int, playerY: int) -> None:
        defaultMap = [
            ["🟥", "⬜️", "⬜️", "🌳", "⬜️", "⬜️", "⬜️", "⬜️", "🟩"],
            ["⬜️", "⬜️", "⬜️", "🌳", "⬜️", "⬜️", "⬜️", "⬜️", "⬜️"],
            ["⬜️", "⬜️", "⬜️", "⬜️", "⬜️", "⬜️", "⬜️", "⬜️", "⬜️"],
            ["⬜️", "🌳", "⬜️", "⬜️", "⬜️", "🌳", "⬜️", "⬜️", "⬜️"],
            ["🟨", "🌳", "⬜️", "⬜️", "⬜️", "🌳", "🟦", "⬜️", "⬜️"],
        ]
        print("0   1   2   3   4")
        for visited in self.visited:
            try:
                defaultMap[visited[0]][visited[1] * 2] = "⬛️"
            except IndexError:
                continue
        defaultMap[playerY][playerX * 2] = "🚕"
        for row in range(defaultMap.__len__()):
            print("".join(defaultMap[row]), end=f"{row}\n")
        print()

    def isInBounds(self, pos: Tuple[int, int]) -> bool:
        return 0 <= pos[0] < self.rows and 0 <= pos[1] < self.cols

    def nextActionDropClient(self) -> int:
        if self.clientDropoff:
            self.clientDropoff = False
            return PICKUP
        if self.taxiPos not in self.visited:
            self.visited.add(self.taxiPos)
            self.clientDropoff = True
            return DROPOFF
        return self.nextActionMove()

    def nextActionGetClient(self) -> int:
        if self.taxiPos not in self.visited:
            self.visited.add(self.taxiPos)
            return PICKUP
        return self.nextActionMove()

    def nextActionMove(self) -> int:
        for action, (dx, dy) in zip([MOVE_DOWN, MOVE_UP, MOVE_RIGHT, MOVE_LEFT], [(1, 0), (-1, 0), (0, 1), (0, -1)]):
            new_pos = (self.taxiPos[0] + dx, self.taxiPos[1] + dy)
            currentSet = frozenset({self.taxiPos, new_pos})

            if self.isInBounds(new_pos) and currentSet not in self.obstacles and new_pos not in self.visited:
                self.fromHsitory = False
                return action

        if self.previous_positions:
            action = self.previous_positions.pop()
            self.fromHsitory = True
            return action

        return -1

    def nextMovePos(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        if action == MOVE_DOWN:
            return (pos[0] + 1, pos[1])
        if action == MOVE_UP:
            return (pos[0] - 1, pos[1])
        if action == MOVE_LEFT:
            return (pos[0], pos[1] - 1)
        if action == MOVE_RIGHT:
            return (pos[0], pos[1] + 1)
        return pos

    def executeAction(self, action: int) -> Tuple[int, int, bool]:
        state, reward, _, truncated, _ = self.env.step(action)
        self.steps += 1
        self.rewards += reward

        lastPos = self.taxiPos
        taxi_row, taxi_col, _, _ = self.env.unwrapped.decode(state)
        self.taxiPos = (taxi_row, taxi_col)

        if action in [MOVE_DOWN, MOVE_UP, MOVE_RIGHT, MOVE_LEFT]:
            if lastPos == self.taxiPos:
                nextMovePosisition = self.nextMovePos(self.taxiPos, action)
                mySet = frozenset({self.taxiPos, nextMovePosisition})
                self.obstacles.add(mySet)
            elif not self.fromHsitory:
                self.previous_positions.append(self.invertDirections[action])

        return state, reward, truncated

    def getClient(self) -> None:
        self.drawMap(self.taxiPos[1], self.taxiPos[0])

        while True:
            action: int = self.nextActionGetClient()
            if action == -1:
                break

            state, reward, truncated = self.executeAction(action)

            taxi_row, taxi_col, _, _ = self.env.unwrapped.decode(state)
            self.taxiPos = (taxi_row, taxi_col)

            # print(f"({taxi_row}, {taxi_col}) {DICT_ACTIONS[action]} {DICT_REWARDS[reward]} {reward} {len(self.visited)}/{self.totalMapPositions}")
            if action < 4:
                self.drawMap(taxi_col, taxi_row)

            if reward == -1 and action == PICKUP:
                print(f"{CYAN}CLIENT RECUPERE{OFF}")
                break

            if len(self.visited) >= self.totalMapPositions:
                break
            if truncated:
                break

    def dropClient(self) -> None:
        self.visited = set()
        self.drawMap(self.taxiPos[1], self.taxiPos[0])

        while True:
            action: int = self.nextActionDropClient()
            if action == -1:
                break

            state, reward, truncated = self.executeAction(action)

            taxi_row, taxi_col, _, _ = self.env.unwrapped.decode(state)
            self.taxiPos = (taxi_row, taxi_col)

            # print(f"({taxi_row}, {taxi_col}) {DICT_ACTIONS[action]} {DICT_REWARDS[reward]} {reward} {len(self.visited)}/{self.totalMapPositions}")
            if action < 4:
                self.drawMap(taxi_col, taxi_row)

            if reward == 20:
                print(f"{CYAN}CLIENT DEPOSE{OFF}")
                break

            if len(self.visited) >= self.totalMapPositions:
                break
            if truncated:
                break

    def drawResult(self) -> None:
        self.env.close()

        self.endTime = time.time()
        print(f"{CYAN}Nombre d'étapes:{OFF} {self.steps}")
        print(f"{CYAN}Rewards:{OFF} {self.rewards}")
        print(f"{CYAN}Temps d'exécution:{OFF} {self.endTime - self.startTime:.2f} secondes")

    def launch(self) -> None:
        self.startTime = time.time()
        self.getClient()
        self.dropClient()
        self.drawResult()
