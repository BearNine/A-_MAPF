import heapq
import matplotlib.pyplot as plt
import numpy as np

class Node:
    """节点类表示搜索树中的每一个点。"""
    def __init__(self, parent=None, position=None):
        self.parent = parent        # 该节点的父节点
        self.position = position    # 节点在迷宫中的坐标位置
        self.g = 0                  # G值：从起点到当前节点的成本
        self.h = 0                  # H值：当前节点到目标点的估计成本
        self.f = 0                  # F值：G值与H值的和，即节点的总评估成本

    # 比较两个节点位置是否相同
    def __eq__(self, other):
        return self.position == other.position

    # 定义小于操作，以便在优先队列中进行比较
    def __lt__(self, other):
        return self.f < other.f
    

def astar(maze, start, end):
    """A*算法实现，用于在迷宫中找到从起点到终点的最短路径。"""
    start_node = Node(None, start)  # 创建起始节点
    end_node = Node(None, end)      # 创建终点节点

    open_list = []                  # 开放列表用于存储待访问的节点
    closed_list = []                # 封闭列表用于存储已访问的节点

    heapq.heappush(open_list, (start_node.f, start_node))  # 将起始节点添加到开放列表
    print("添加起始节点到开放列表。")

    # 当开放列表非空时，循环执行
    while open_list:
        current_node = heapq.heappop(open_list)[1]  # 弹出并返回开放列表中 f 值最小的节点
        closed_list.append(current_node)            # 将当前节点添加到封闭列表
        print(f"当前节点: {current_node.position}")

        # 如果当前节点是目标节点，则回溯路径
        if current_node == end_node:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            print("找到目标节点，返回路径。")
            return path[::-1]  # 返回反向路径，即从起点到终点的路径

        # 获取当前节点周围的相邻节点
        (x, y) = current_node.position
        neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]

        # 遍历相邻节点
        for next in neighbors:
            # 确保相邻节点在迷宫范围内，且不是障碍物
            if 0 <= next[0] < maze.shape[0] and 0 <= next[1] < maze.shape[1]:
                if maze[next[0], next[1]] == 1:
                    continue
                neighbor = Node(current_node, next)  # 创建相邻节点
                # 如果相邻节点已在封闭列表中，跳过不处理
                if neighbor in closed_list:
                    continue
                neighbor.g = current_node.g + 1  # 计算相邻节点的 G 值
                neighbor.h = ((end_node.position[0] - next[0]) ** 2) + ((end_node.position[1] - next[1]) ** 2)  # 计算 H 值
                neighbor.f = neighbor.g + neighbor.h  # 计算 F 值

                # 如果相邻节点的新 F 值较小，则将其添加到开放列表
                if add_to_open(open_list, neighbor):
                    heapq.heappush(open_list, (neighbor.f, neighbor))
                    print(f"添加节点 {neighbor.position} 到开放列表。")
            else:
                print(f"节点 {next} 越界或为障碍。")

    return None  # 如果没有找到路径，返回 None

def add_to_open(open_list, neighbor):
    """检查并添加节点到开放列表。"""
    for node in open_list:
        # 如果开放列表中已存在相同位置的节点且 G 值更低，不添加该节点
        if neighbor == node[1] and neighbor.g > node[1].g:
            return False
    return True  # 如果不存在，则返回 True 以便添加该节点到开放列表

def visualize_path(maze, astar_path, ql_path, start, end):
    """将两种算法的路径可视化在迷宫上。"""
    maze_copy = np.array(maze)
    plt.figure(figsize=(12, 12))
    
    # 绘制迷宫
    plt.imshow(maze_copy, cmap='hot', interpolation='nearest')
    
    # 绘制A*路径
    if astar_path:
        astar_x = [p[1] for p in astar_path]
        astar_y = [p[0] for p in astar_path]
        plt.plot(astar_x, astar_y, color='orange', linewidth=3, label='A* Path')
    
    # 绘制Q-learning路径
    if ql_path:
        ql_x = [p[1] for p in ql_path]
        ql_y = [p[0] for p in ql_path]
        plt.plot(ql_x, ql_y, color='blue', linestyle='--', linewidth=3, label='Q-learning Path')
    
    # 绘制起点和终点
    start_x, start_y = start[1], start[0]
    end_x, end_y = end[1], end[0]
    plt.scatter([start_x], [start_y], color='green', s=200, label='Start', zorder=5)
    plt.scatter([end_x], [end_y], color='red', s=200, label='End', zorder=5)
    
    # 添加图例和标题
    plt.legend(fontsize=12)
    plt.title('Path Planning Comparison', fontsize=16)
    plt.show()

# 设定迷宫的尺寸
maze_size = 100

# 创建一个空的迷宫，全部设置为0（表示可通过）
maze = np.zeros((maze_size, maze_size))

# 定义几个障碍物区块，每个障碍物区块是一个矩形
obstacle_blocks = [
    (10, 10, 20, 20),  # (y起始, x起始, 高度, 宽度)
    (30, 40, 20, 30),
    (60, 20, 15, 10),
    (80, 50, 10, 45),
    (60, 60, 10, 10),
    (50, 10, 3, 85),
]

# 在迷宫中设置障碍物
for y_start, x_start, height, width in obstacle_blocks:
    maze[y_start:y_start+height, x_start:x_start+width] = 1

# 设定起始点和终点
start = (0, 0)
end = (92, 93)

# 确保起始点和终点不是障碍物
maze[start] = 0
maze[end] = 0

# 输出迷宫的一部分，以确认障碍物的设置
print("迷宫左上角10x10区域的视图:")
print(maze[:10, :10])

# Q-learning参数优化
alpha = 0.2    # 提高学习率加速收敛
gamma = 0.9    # 降低折扣因子更关注近期奖励
epsilon = 1.0   # 初始探索率
epsilon_decay = 0.995  # 探索率衰减系数
min_epsilon = 0.01     # 最小探索率
episodes = 2000        # 增加训练轮数

# 初始化Q表
Q = {}

# 动作空间：上(0)、下(1)、左(2)、右(3)
actions = [0, 1, 2, 3]

def get_state(state):
    """标准化状态表示"""
    return tuple(state)

def choose_action(state):
    """改进的ε-greedy策略选择动作"""
    global epsilon  # 声明全局变量
    state_key = get_state(state)
    if np.random.uniform(0, 1) < epsilon:
        # 在探索时优先选择朝向目标的方向
        dx = end[0] - state[0]
        dy = end[1] - state[1]
        preferred_actions = []
        if dx > 0: preferred_actions.append(1)  # 向下
        elif dx < 0: preferred_actions.append(0)  # 向上
        if dy > 0: preferred_actions.append(3)  # 向右
        elif dy < 0: preferred_actions.append(2)  # 向左
        
        if preferred_actions:
            return np.random.choice(preferred_actions)
    else:
        # 获取当前状态的所有Q值
        q_values = Q.get(state_key, np.zeros(len(actions)))
        return np.argmax(q_values)  # 利用

def update_q_value(state, action, reward, next_state):
    """更新Q值"""
    state_key = get_state(state)
    next_state_key = get_state(next_state)
    
    # 初始化Q值
    if state_key not in Q:
        Q[state_key] = np.zeros(len(actions))
    if next_state_key not in Q:
        Q[next_state_key] = np.zeros(len(actions))
        
    # Q-learning更新公式
    Q[state_key][action] = (1 - alpha) * Q[state_key][action] + alpha * (
        reward + gamma * np.max(Q[next_state_key])
    )

def get_reward(next_pos, end):
    """改进的奖励函数"""
    # 到达终点的奖励
    if next_pos == end:
        return 500
    
    # 碰撞障碍物惩罚
    if maze[next_pos] == 1:
        return -200
    
    # 计算曼哈顿距离进步
    current_dist = abs(next_pos[0]-end[0]) + abs(next_pos[1]-end[1])
    last_dist = abs(next_pos[0]-end[0]) + abs(next_pos[1]-end[1]) + 1  # 假设上一步更远
    
    # 基础步长惩罚 + 距离进步奖励
    distance_reward = (last_dist - current_dist) * 5  # 每接近一步奖励5
    return -1 + distance_reward

def q_learning_train(maze, start, end):
    """Q-learning训练函数"""
    global epsilon  # 声明全局变量
    for episode in range(episodes):
        # 衰减探索率
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        state = start
        total_reward = 0
        steps = 0
        max_steps = 1000
        
        while state != end and steps < max_steps:
            action = choose_action(state)
            
            # 执行动作
            next_state = list(state)
            if action == 0:   # 上
                next_state[0] -= 1
            elif action == 1: # 下
                next_state[0] += 1
            elif action == 2: # 左
                next_state[1] -= 1
            elif action == 3: # 右
                next_state[1] += 1
                
            next_state = tuple(next_state)
            
            # 边界检查
            if (next_state[0] < 0 or next_state[0] >= maze.shape[0] or 
                next_state[1] < 0 or next_state[1] >= maze.shape[1]):
                next_state = state  # 保持原位
                reward = -10        # 越界惩罚
            else:
                reward = get_reward(next_state, end)
            
            # 更新Q值
            update_q_value(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if maze[state] == 1:  # 如果进入障碍物，重置位置
                state = start
                
        if (episode+1) % 50 == 0:
            print(f"Episode: {episode+1}, Total Reward: {total_reward:.2f}")

def extract_ql_path(start, end):
    """改进的路径提取函数"""
    state = start
    path = [state]
    visited = set()
    max_steps = 500  # 最大步数限制
    step = 0
    
    while state != end and step < max_steps:
        state_key = get_state(state)
        if state_key not in Q:
            # 尝试随机移动避免卡死
            action = np.random.choice(actions)
        else:
            # 添加10%随机性避免局部最优
            if np.random.rand() < 0.1:
                action = np.random.choice(actions)
            else:
                action = np.argmax(Q[state_key])
        
        next_state = list(state)
        if action == 0:   # 上
            next_state[0] -= 1
        elif action == 1: # 下
            next_state[0] += 1
        elif action == 2: # 左
            next_state[1] -= 1
        elif action == 3: # 右
            next_state[1] += 1
            
        next_state = tuple(next_state)
        
        # 边界和障碍检查
        if (next_state[0] < 0 or next_state[0] >= maze.shape[0] or 
            next_state[1] < 0 or next_state[1] >= maze.shape[1] or 
            maze[next_state] == 1):
            # 遇到无效位置时尝试随机方向
            next_state = state
            action = np.random.choice(actions)
            continue
            
        # 检查循环，如果重复访问则回溯两步
        if next_state in visited:
            if len(path) >= 2:
                path.pop()
                state = path[-1]
            continue
        visited.add(next_state)
        
        path.append(next_state)
        state = next_state
        step += 1
        
    return path if state == end else path  # 返回已找到的最佳路径

# 运行A*算法
astar_path = astar(maze, start, end)

# 训练Q-learning算法
print("\n开始Q-learning训练...")
# q_learning_train(maze, start, end)

# 提取Q-learning路径
# ql_path = extract_ql_path(start, end)

# 可视化结果
print("\nA*路径长度:", len(astar_path) if astar_path else 0)
# print("Q-learning路径长度:", len(ql_path) if ql_path else 0)
# visualize_path(maze, astar_path, ql_path, start, end)
