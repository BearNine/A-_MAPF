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

    # 当开放列表非空时，循环执行（增加最大步数限制）
    max_steps = maze.shape[0] * maze.shape[1] * 10  # 根据迷宫大小动态计算最大步数
    step_counter = 0
    
    while open_list:
        step_counter += 1
        if step_counter > max_steps:
            print("警告：超过最大搜索步数，终止搜索！")
            return None
            
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

def visualize_q_table(maze, Q, start, end):
    """可视化Q表中每个状态的最大Q值对应动作方向"""
    plt.figure(figsize=(16, 16))
    plt.imshow(maze, cmap='hot', interpolation='nearest')
    
    # 增强箭头可视化参数
    arrow_style = {
        0: {'dx':0, 'dy':-0.45, 'color':'#FFFF00', 'width':0.05, 'head_width':0.3, 'label':'Up'},     # 亮黄色-上
        1: {'dx':0, 'dy':0.45, 'color':'#00FF00', 'width':0.05, 'head_width':0.3, 'label':'Down'},    # 亮绿色-下
        2: {'dx':-0.45, 'dy':0, 'color':'#00FFFF', 'width':0.05, 'head_width':0.3, 'label':'Left'},   # 亮青色-左
        3: {'dx':0.45, 'dy':0, 'color':'#FF00FF', 'width':0.05, 'head_width':0.3, 'label':'Right'}    # 亮品红-右
    }

    # 遍历迷宫中的每个格子
    for y in range(maze.shape[0]):
        for x in range(maze.shape[1]):
            if maze[y, x] == 1:  # 跳过障碍物
                continue
            if (y, x) == start or (y, x) == end:  # 跳过起点终点
                continue
                
            state = (y, x)
            if state in Q:
                # 获取最大Q值对应的动作
                action = np.argmax(Q[state])
                # 绘制箭头
                params = arrow_style[action]
                plt.arrow(x, y, params['dx'], params['dy'], 
                          color=params['color'], width=params['width'],
                          head_width=params['head_width'], length_includes_head=True)

    # 绘制起点和终点
    plt.scatter(start[1], start[0], color='green', s=200, label='Start', zorder=5)
    plt.scatter(end[1], end[0], color='red', s=200, label='End', zorder=5)
    
    plt.title('Q-table Policy Visualization', fontsize=16)
    plt.legend()
    plt.show()

def visualize_path(maze, astar_path, ql_path, start, end):
    """将两种算法的路径可视化在迷宫上。"""
    maze_copy = np.array(maze)
    plt.figure(figsize=(16, 16))
    
    # 使用更清晰的配色方案
    cmap = plt.cm.get_cmap('viridis').copy()
    cmap.set_under('white')  # 可通行区域
    cmap.set_over('black')   # 障碍物
    
    # 绘制迷宫
    plt.imshow(maze_copy, cmap=cmap, interpolation='nearest', vmin=0, vmax=1)
    
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
    plt.title('路径规划算法对比 (A* vs Q-learning)', fontsize=18)
    plt.xlabel('X 坐标', fontsize=14)
    plt.ylabel('Y 坐标', fontsize=14)
    
    # 添加性能对比标注
    text_str = f"""A* 路径长度: {len(astar_path) if astar_path else 0}
Q-learning 路径长度: {len(ql_path) if ql_path else 0}
迷宫尺寸: {maze_size}x{maze_size}"""
    plt.text(0.5, -0.1, text_str, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', horizontalalignment='center',
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
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
    (80, 50, 5, 45),
    (60, 60, 10, 10),
    (50, 10, 3, 85),
    (60, 80, 3, 25),
]

# 在迷宫中设置障碍物
for y_start, x_start, height, width in obstacle_blocks:
    maze[y_start:y_start+height, x_start:x_start+width] = 1

# 设定起始点和终点
start = (0, 0)
end = (90, 90)

# 确保起始点和终点不是障碍物
maze[start] = 0
maze[end] = 0

# 输出迷宫的一部分，以确认障碍物的设置
print("迷宫左上角10x10区域的视图:")
print(maze[:10, :10])

# Q-learning参数优化（强化版本）
alpha = 0.3      # 提高学习率加速收敛
gamma = 0.95     # 平衡远期奖励
epsilon = 1.0     # 初始探索率
epsilon_decay = 0.999  # 更平缓的探索率衰减
min_epsilon = 0.1    # 保持最小探索率
episodes = 5000       # 增加训练轮数
batch_size = 10      # 每10轮显示一次进度

# 初始化Q表
Q = {}

# 动作空间：上(0)、下(1)、左(2)、右(3)
actions = [0, 1, 2, 3]

def get_state(state):
    """标准化状态表示"""
    return tuple(state)

# 添加在文件开头
from collections import deque

# 在全局变量区域添加
visited_states = deque(maxlen=10)  # 记录最近10个状态
direction_weights = {0:1.0, 1:1.0, 2:1.0, 3:1.0}  # 方向基础权重

def get_next_state(state, action):
    """计算执行动作后的下一个状态"""
    y, x = state
    if action == 0:   # 上
        y -= 1
    elif action == 1: # 下
        y += 1
    elif action == 2: # 左
        x -= 1
    elif action == 3: # 右
        x += 1
    return (y, x)

def get_freshness_score(state, action):
    """计算动作的新鲜度评分，避免重复路径"""
    next_state = get_next_state(state, action)
    repeat_penalty = 0.3  # 重复状态的惩罚系数
    base_score = 1.0
    
    # 计算该状态在历史中出现的次数
    repeat_count = list(visited_states).count(next_state)
    return base_score * (repeat_penalty ** repeat_count)

def choose_action(state):
    """改进的ε-greedy策略选择动作（带路径防重复）"""
    global epsilon
    state_key = get_state(state)
    
    # 记录当前状态
    visited_states.append(state)
    
    if np.random.uniform(0, 1) < epsilon:
        dx = end[0] - state[0]
        dy = end[1] - state[1]
        action_scores = []
        
        # 计算每个动作的吸引力分数
        for action in actions:
            # 基础方向偏好
            dir_score = 0
            if (action == 1 and dx > 0) or (action == 0 and dx < 0):
                dir_score += 1.5
            if (action == 3 and dy > 0) or (action == 2 and dy < 0):
                dir_score += 1.5
                
            # 路径新鲜度评分
            freshness = get_freshness_score(state, action)
            
            # 综合得分 = 方向偏好 + 新鲜度
            action_scores.append(dir_score + freshness)
        
        # 根据得分概率选择动作
        scores = np.array(action_scores)
        prob = scores / scores.sum()
        return np.random.choice(actions, p=prob)
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

def get_reward(current_pos, next_pos, end):
    """改进的奖励函数"""
    # 到达终点的奖励
    if next_pos == end:
        return 500000
    
    # 碰撞障碍物惩罚
    if maze[next_pos] == 1:
        return -200
    
    # 计算几何距离进步（欧几里得距离平方）
    current_dist = (next_pos[0]-end[0])**2 + (next_pos[1]-end[1])**2
    last_dist = (current_pos[0]-end[0])**2 + (current_pos[1]-end[1])**2  # 使用实际的上一步距离
    
    # 基础步长惩罚 + 真实距离变化奖励
    distance_reward = (last_dist - current_dist) * 0.1  # 实际距离变化的奖励
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
        max_steps = 5000
        
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
                reward = get_reward(state, next_state, end)
            
            # 更新Q值
            update_q_value(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if maze[state] == 1:  # 如果进入障碍物，重置位置
                state = start
                
        if (episode+1) % batch_size == 0:
            print(f"Episode: {episode+1}, Total Reward: {total_reward:.2f}")

def extract_ql_path(start, end):
    """改进的路径提取函数（增强版）"""
    state = start
    path = [state]
    visited = {}
    max_steps = 1000  # 增大步数限制
    step = 0
    last_good_position = start  # 记录最后有效位置
    
    while state != end and step < max_steps:
        state_key = get_state(state)
        
        # 动态调整探索率（随步数增加逐渐降低）
        current_epsilon = max(0.01, epsilon * (1 - step/max_steps))
        
        if state_key not in Q or np.random.rand() < current_epsilon:
            # 优先选择朝向目标的方向
            dx = end[0] - state[0]
            dy = end[1] - state[1]
            preferred_actions = []
            if dx > 0: preferred_actions.append(1)  # 下
            elif dx < 0: preferred_actions.append(0)  # 上
            if dy > 0: preferred_actions.append(3)  # 右
            elif dy < 0: preferred_actions.append(2)  # 左
            
            if preferred_actions:
                action = np.random.choice(preferred_actions)
            else:
                action = np.random.choice(actions)
        else:
            # 添加基于Q值的概率选择（softmax策略）
            q_values = Q[state_key]
            exp_q = np.exp(q_values - np.max(q_values))
            probabilities = exp_q / (exp_q.sum() + 1e-10)  # 添加极小值防止除零
            action = np.random.choice(actions, p=probabilities)
        
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
        
        # 边界和障碍检查（增强版）
        if (next_state[0] < 0 or next_state[0] >= maze.shape[0] or 
            next_state[1] < 0 or next_state[1] >= maze.shape[1] or 
            maze[next_state] == 1):
            # 遇到无效位置时进行惩罚并尝试新方向
            if np.random.rand() < 0.5:
                next_state = last_good_position  # 回到最后有效位置
            else:
                next_state = state
            continue
        
        # 更新最后有效位置
        last_good_position = state
        
        # 改进循环检测机制
        if next_state in visited:
            # 如果循环超过3次，回溯到循环起点
            if visited[next_state] > 3 and len(path) > 10:
                path = path[:-5]  # 回溯5步
                state = path[-1]
                continue
            visited[next_state] += 1
        else:
            visited[next_state] = 1
            
        path.append(next_state)
        state = next_state
        step += 1
        
        # 动态调整：每100步重置部分visited记录
        if step % 100 == 0:
            visited = {k: v for k, v in visited.items() if v > 2}
            
    return path

# 运行A*算法
astar_path = astar(maze, start, end)

# 训练Q-learning算法
print("\n开始Q-learning训练...")
q_learning_train(maze, start, end)

# 可视化Q表策略
print("\n正在可视化Q-learning策略...")
visualize_q_table(maze, Q, start, end)

# 提取Q-learning路径
ql_path = extract_ql_path(start, end)

# 可视化对比结果
print("\nA*路径长度:", len(astar_path) if astar_path else 0)
print("Q-learning路径长度:", len(ql_path) if ql_path else 0)
visualize_path(maze, astar_path, ql_path, start, end)
