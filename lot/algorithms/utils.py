import re
import pandas as pd

def clean_str(p):
    return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")

def replace_percentages(text):
    pattern = r'(\d+)%'
    
    def replace_match(match):
        number = int(match.group(1))
        return str(number / 100)

    result = re.sub(pattern, replace_match, text)
    return result

def split_str(strs):
    sent_str = strs.split("。")
    all_strs = ''
    for sent in sent_str:
        piece_str = sent.split(",")
        for piece in piece_str:
            all_strs = all_strs + piece + '\n'
    return all_strs

def tree_walk(root, data, task):
    if task.mode == 'mcts':
        for child in root.children.values():
            trans_str = split_str(child.pcd)
            str2 = trans_str + '\nAccess sequence: ' + str(child.visit_sequence) + '\nValue: ' + str(child.V) + '\nflag: ' + str(child.final_ans_flag)
            data['id'].append(child.visit_sequence)
            data['parent'].append(root.visit_sequence)
            data['value'].append(child.V)
            data['sequence'].append(str2)
            tree_walk(child, data, task)
    else:
        for child in root.children:
            trans_str = split_str(child.pcd)
            str2 = trans_str + '\nAccess sequence: ' + str(child.visit_sequence) + '\nValue: ' + str(child.V) + '\nflag: ' + str(
                child.final_ans_flag)
            data['id'].append(child.visit_sequence)
            data['parent'].append(root.visit_sequence)
            data['value'].append(child.V)
            data['sequence'].append(str2)
            tree_walk(child, data, task)

def get_tree(task, root):
    data = {
        'id': [],
        'sequence': [],
        'parent': [],
        'value': [],
    }

    str1 = 'Question: ' + split_str(task.question) + '\nAccess sequence: ' + str(root.visit_sequence) + '\nValue: ' + str(
        root.V) + '\nflag: ' + str(root.final_ans_flag)
    data['id'].append(root.visit_sequence)
    data['parent'].append(None)
    data['value'].append(-1)
    data['sequence'].append(str1)
    tree_walk(root, data, task)

    df = pd.DataFrame(data).sort_values(by=['id'])
    df.reset_index(drop=True, inplace=True)

    return df