import time
import math
import random
import numpy
from functools import partial
import copy
from .base import treeNode


def get_next_steps_roll(y: str, step_n: int, mcts_task):
    next_steps = []
    for i in range(mcts_task.roll_branch):
        proposal = ''
        cnt = 3
        while not proposal and cnt:
            proposal = mcts_task.get_next_step(y, step_n)
            cnt -= 1
        if not proposal:
            continue
        next_steps.append(proposal)
    return next_steps


def get_next_steps_expand(node: treeNode, mcts_task):
    next_steps = []
    reflection = node.reflection
    for i in range(mcts_task.branch):
        proposal = ''
        cnt = 3
        while not proposal and cnt:
            if mcts_task.use_reflection == 'common':
                proposal = mcts_task.get_next_step_use_reflection(node.y, node.depth + 1, reflection)
            else:
                proposal = mcts_task.get_next_step(node.y, node.depth + 1)
            cnt -= 1
        if not proposal:
            continue
        next_steps.append(proposal)
    return next_steps


def randomPolicy(node: treeNode, mcts_task):
    max_V = mcts_task.low
    strs = node.y
    cur_step = node.depth + 1
    if mcts_task.use_reflection == 'common':
        reflection = mcts_task.get_reflection(strs, cur_step)
    else:
        reflection = mcts_task.get_simple_reflection(strs, cur_step)
    node.update_reflection(reflection)
    if reflection == '<end>':
        print('This step has been resolved and does not require simulation.\n')
        return node.V
    for i in range(mcts_task.roll_forward_steps):
        next_steps = get_next_steps_roll(strs, cur_step, mcts_task)
        if not next_steps:
            break
        action = random.choice(next_steps)  # str
        strs = strs + action
        cur_step += 1
        value = mcts_task.get_step_value(strs)
        if value > max_V:
            max_V = value
        if mcts_task.use_reflection == 'common':
            cur_ref = mcts_task.get_reflection(strs, cur_step)
        else:
            cur_ref = mcts_task.get_simple_reflection(strs, cur_step)
        if cur_ref == '<end>':
            break
    return max_V


def greedyPolicy(node: treeNode, mcts_task):
    max_V = mcts_task.low
    strs = node.y
    cur_step = node.depth + 1
    if mcts_task.use_reflection == 'common':
        reflection = mcts_task.get_reflection(strs, cur_step)
    else:
        reflection = mcts_task.get_simple_reflection(strs, cur_step)
    node.update_reflection(reflection)
    if reflection == '<end>':
        print('This step has been resolved and does not require simulation.\n')
        return node.V
    for i in range(mcts_task.roll_forward_steps):
        actions = get_next_steps_roll(strs, cur_step, mcts_task)  # str_list
        if not actions:
            break
        new_ys = [strs + action for action in actions]
        cur_step += 1
        values = [mcts_task.get_step_value(new_y) for new_y in new_ys]
        idx = numpy.argmax(values)
        strs = new_ys[idx]
        value = values[idx]
        if value > max_V:
            max_V = value
        if mcts_task.use_reflection == 'common':
            cur_ref = mcts_task.get_reflection(strs, cur_step)
        else:
            cur_ref = mcts_task.get_simple_reflection(strs, cur_step)
        if cur_ref == '<end>':
            break
    return max_V


def MCTS_search(mcts_task):
    root = treeNode('')

    if mcts_task.limit_type == 'time':
        timeLimit = time.time() + mcts_task.time_limit / 1000
        time_start = time.time()
        while time.time() < timeLimit:
            flag, node, root = executeRound(root, mcts_task)
            if flag:
                return root, node, time.time() - time_start
    else:
        for i in range(mcts_task.iteration_limit):
            flag, node, root = executeRound(root, mcts_task)
            if flag:
                return root, node, i + 1
    return root, None, None


def executeRound(root, mcts_task):
    # execute a selection-expansion-simulation-backpropagation round

    flag, node = selectNode(root, mcts_task)
    if flag:
        if mcts_task.sample_value != 'full':
            return True, node, root
        else:
            node.reflection = '<end>'

    if node.reflection == '<end>':
        print('Skip this round.\n')
    else:
        node = expand(node, mcts_task)

    if mcts_task.reward_model_type == 'vm':
        if node.reflection == '<end>':
            print('Skip this round.\n')
        else:
            roll_node = getBestChild(node, mcts_task)
            best_V = greedyPolicy(roll_node, mcts_task) if mcts_task.roll_policy == 'greedy' else randomPolicy(roll_node, mcts_task)
            roll_node.V = roll_node.V * (1 - mcts_task.alpha) + best_V * mcts_task.alpha
            roll_node.numVisits += 1

    back_propagate(node)
    return False, node, root


def isTerminal(node, mcts_task):
    if mcts_task.reward_model_type == 'vm':
        return node.V >= mcts_task.end_gate
    else:
        return False


def selectNode(node, mcts_task):
    while node.isFullyExpanded:
        node = getBestChild(node, mcts_task)
    if isTerminal(node, mcts_task):
        node.final_ans_flag = 1
        return True, node
    else:
        return False, node


def expand(node: treeNode, mcts_task):
    if not node.reflection:
        if mcts_task.use_reflection == 'common':
            reflection = mcts_task.get_reflection(node.y, node.depth + 1)
        else:  # simple
            reflection = mcts_task.get_simple_reflection(node.y, node.depth + 1)
        node.update_reflection(reflection)
    if node.reflection == '<end>':
        return node
    actions = get_next_steps_expand(node, mcts_task)
    if not actions:
        node.update_reflection('<end>')
        return node

    for action in actions:
        if action not in node.children.keys():
            node.append_children(action)
            child = node.children[action]
            value = mcts_task.get_step_value(child.y)
            child.update_value(value)
            if mcts_task.sample_value == 'full':
                if mcts_task.use_reflection == 'common':
                    child.update_reflection(mcts_task.get_reflection(child.y, child.depth + 1))
                else:
                    child.update_reflection(mcts_task.get_simple_reflection(child.y, child.depth + 1))
            child.visit_sequence = mcts_task.node_count
            mcts_task.update_count()
    node.isFullyExpanded = True
    return node


def back_propagate(node):
    while node is not None:
        node.numVisits += 1
        if node.isFullyExpanded:
            child_Vs = [child.V * child.numVisits for child in node.children.values()]
            total_num_visits = sum([child.numVisits for child in node.children.values()])
            if total_num_visits > 0:
                node.V = sum(child_Vs) / total_num_visits
        node = node.parent


def getBestChild(node, mcts_task):
    bestValue = mcts_task.low
    bestNodes = []
    for child in node.children.values():
        nodeValue = child.V + mcts_task.exploration_constant * math.sqrt(
            2 * math.log(node.numVisits) / child.numVisits) if child.numVisits > 0 else child.V + mcts_task.INF
        if nodeValue > bestValue:
            bestValue = nodeValue
            bestNodes = [child]
        elif nodeValue == bestValue:
            bestNodes.append(child)
    return random.choice(bestNodes)


def MCTS(mcts_task):
    root, node, finish = MCTS_search(mcts_task)

    if mcts_task.sample_value == 'full':
        print('Sampling completed.\n')
        return None, -1, root
    else:
        if mcts_task.reward_model_type == 'vm':
            if finish is not None:
                print(f'Solution found!\nSolution:{node.y}\n')
                return node, finish, root

            else:
                best_node, best_V = root.getBestV()
                print(f'No solution found within the specified time/rounds, using the highest value solution instead.\nSolution:{best_node.y}\n')
                return best_node, -1, root
        else:
            print('Sampling completed.\n')
            return None, -1, root
