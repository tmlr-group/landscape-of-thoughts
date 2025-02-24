import copy
import json
import os
import pickle as pkl
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import umap.umap_ as umap
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm, trange
from xgboost import XGBClassifier

'''
    File --> Sample data
'''
def load_chain_data_and_plot(thoughts_file: str = "None",
        tool: str = 'tsne',
        plot: bool = False,
):

    # load data
    #######################################
    assert os.path.exists(thoughts_file), print(thoughts_file)
    trial_data = json.load(open(thoughts_file, 'r'))
    model_input = trial_data["model_input"]
    answers = trial_data["answers"]
    answer_gt_full = trial_data["answer_gt_full"]
    answer_gt_short = trial_data["answer_gt_short"]
    trial_thoughts = trial_data["trial_thoughts"]
    pkl_path = thoughts_file.replace(".json", f".pkl")
    pkl_path = pkl_path.replace("thoughts/", "distance_matrix/")  
    assert os.path.exists(pkl_path)
    distance_matrix = pkl.load(open(pkl_path, 'rb'))

    # pre-process for the visualize
    #######################################
    chain_color = []
    labels = []
    chain_corr = []
    for [_, answer, binary_label] in trial_thoughts:
        if binary_label == True:
            chain_color.append("green")
            chain_corr.append(binary_label)
        else:
            chain_color.append("red")
            chain_corr.append(binary_label)
        labels.append(binary_label)

    # parse thoughts
    #######################################
    num_chains = len(trial_thoughts)
    num_thoughts_each_chain = [len(thoughts) for [thoughts, answer, binary_label] in trial_thoughts]
    all_answers = [answer for [thoughts, answer, binary_label] in trial_thoughts]
    all_thoughts = []
    for [thoughts, answer, binary_label] in trial_thoughts:
        all_thoughts += thoughts
    all_thoughts = np.array(all_thoughts)
    num_all_thoughts = len(all_thoughts)
    anchors = [model_input] + answers # question and answers
    num_anchors = len(anchors)
    anchors_idx_y = [i for i in range(num_anchors)]
    anchors_idx_x = [(num_all_thoughts + i) for i in range(num_anchors)]

    # draw the landscape
    #######################################    
    if "strategyqa" in thoughts_file:
        labels_anchors = ["Start", 'A', 'B']
        if answer_gt_short:
            answer_gt_short = 'A' # yes
        else:
            answer_gt_short = 'B' # no
        gt_idx = labels_anchors.index(answer_gt_short)
    else:
        labels_anchors = ["Start", 'A', 'B', 'C', 'D', 'E']
        gt_idx = labels_anchors.index(answer_gt_short)
    answer_idx_y = anchors_idx_y[gt_idx] # the groud truth answer

    # # normlize thought (T matrix)  
    processed_distance_matrix = copy.deepcopy(distance_matrix)  
    # ######################################################
    
    for chain_idx in range(num_chains): 
        start_idx, end_idx = sum(num_thoughts_each_chain[:chain_idx]), sum(num_thoughts_each_chain[:chain_idx+1])
        accumulate_ppl = 0
        for thought_idx in range(start_idx, end_idx):
            accumulate_ppl += distance_matrix[thought_idx, 0]
            # norm D(X, T)
            processed_distance_matrix[thought_idx, 0] = accumulate_ppl / np.sum(distance_matrix[start_idx:end_idx, 0])
            # norm D(T, Y)
            for anchor_idx in range(1, num_anchors):
                processed_distance_matrix[thought_idx, anchor_idx] = distance_matrix[thought_idx, anchor_idx] / np.sum(distance_matrix[thought_idx, anchors_idx_y[1:]])

    # check the normalize effect
    for chain_idx in range(num_chains): 
        start_idx, end_idx = sum(num_thoughts_each_chain[:chain_idx]), sum(num_thoughts_each_chain[:chain_idx+1])
        # assert np.abs(np.sum(processed_distance_matrix[start_idx:end_idx, 0])- 1) < 1e-5 # first col, each chian sum would be 1
        for thought_idx in range(start_idx, end_idx): # row-sum=1
            assert ( (np.sum(processed_distance_matrix[start_idx:end_idx, anchors_idx_y[1:]], axis=1) - 1) < 1e-5 ).all()

    # normlize answer (A matrix)
    ######################################################
    A = processed_distance_matrix[num_all_thoughts:]
    A[np.diag_indices(A.shape[0])] = 0
    normed_A = copy.deepcopy(A)
    for col_idx in range(1, num_anchors):
        normed_A[0][col_idx] = 1 + A[0][col_idx] / np.sum(A[0, anchors_idx_y[1:]])
    normed_A[:, 0] = normed_A[0, :] # copy same elements to 0-th col
    normed_A[1:, 1:] = 1 / (num_anchors-1) 
    normed_A[np.diag_indices(normed_A.shape[0])] = 0
    processed_distance_matrix[num_all_thoughts:] = normed_A

    distance_matrix = processed_distance_matrix

    coordinates_value = distance_matrix[:, answer_idx_y] / np.sum(distance_matrix[:, anchors_idx_y[1:]], axis=1) # v2 normalize
    if tool == 'tsne':
        tsne = TSNE(n_components=2, perplexity=10, random_state=42)
        coordinates_2d = tsne.fit_transform(distance_matrix[:, 1:])
    elif tool == 'umap':
        coordinates_2d = umap.UMAP(n_neighbors=30, min_dist=0.25, n_components=2, metric='dice', random_state=42).fit_transform(distance_matrix)

    if plot:
        # plot states
        plt.figure(figsize=(8, 6))    
        fig = go.Figure()

        # Not using the z value
        fig = go.Figure(
            data = go.Histogram2dContour(
                x = coordinates_2d[:, 0],
                y = coordinates_2d[:, 1],
                colorscale = 'viridis',
                showlegend=False,
                contours = dict(
                    showlabels = True,
                    coloring='none',  # Turn off the fill between contour lines
                    labelfont = dict(
                        family = 'Raleway',
                        color = 'black',
                        size=20
                        )
                    )
            ))
        
        colors = px.colors.qualitative.Light24
        for chain_idx in range(num_chains):
            start_idx, end_idx = sum(num_thoughts_each_chain[:chain_idx]), sum(num_thoughts_each_chain[:chain_idx+1])
            x, y = list(coordinates_2d[start_idx:end_idx, 0]), list(coordinates_2d[start_idx:end_idx, 1])

            # Normalized indices to use for the color scale
            normalized_indices = np.linspace(0, 1, len(x))

            # Create the scatter plot for the original xy points
            fig.add_trace(
                go.Scatter(x=x, y=y,
                                mode='markers', 
                                marker_symbol='diamond' if chain_corr[chain_idx] else 'x',
                                marker_size=10,
                                marker=dict(
                                    size=5, 
                                    color=normalized_indices,  # Color based on index within the chain
                                    colorscale='RdYlGn',  # Choose a colorscale (Viridis is an example)
                                    showscale=True,
                                ), # green or red
                                showlegend=False,
                                line_color='black',
                                customdata=[[chain_idx, round(coordinates_value[start_idx+thought_idx], 3), all_thoughts[start_idx+thought_idx]] for thought_idx in range(len(x))],
                                hovertemplate=
                                "<b>Chain-%{customdata[0]}</b><br>"   # Chain index
                                +"PPL: %{customdata[1]}<br>"      # Average PPL
                                # +"Thought: %{customdata[2]}<br>"     # Thought 
                                +"X: %{x}<br>"                       # X value
                                +"Y: %{y}<br>"                         # Y value
                        ))

        # Plot anchors
        #####################################
        colors = px.colors.qualitative.Plotly
        for idx, anchor_name in enumerate(labels_anchors):
            if anchor_name == 'Start': # start
                marker_symbol = 'star'
            elif anchor_name == answer_gt_short: 
                # correct answer
                marker_symbol='diamond'
            else: # negative answer 
                marker_symbol = 'x'

            fig.add_trace(
                go.Scatter(
                    x=[coordinates_2d[anchors_idx_x[idx], 0]], y=[coordinates_2d[anchors_idx_x[idx], 1]], 
                    mode='markers',
                    name=anchor_name, marker=dict(symbol=marker_symbol, size=20, line_width=1, color=colors[idx % len(colors)])
            ))

        fig.update_layout(
            legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.01, font=dict(size=20)), 
            margin=dict(l=10, r=10, t=20, b=10),
        )
        fig.show()

    return distance_matrix, num_chains, num_thoughts_each_chain, coordinates_2d, normed_A, all_answers, answer_gt_short, coordinates_2d[anchors_idx_x[gt_idx], :]

def load_sample_data(
    model = 'Meta-Llama-3.1-70B-Instruct-Turbo',
    dataset = 'aqua',
    method = "zero_shot_cot",
    tool = 'tsne',
    start_sample_idx = 0,
    end_sample_idx = 10
):
    list_distance_matrix = []
    list_num_chains = []
    list_num_thoughts_each_chain = []
    list_coordinates_2d = []
    list_normed_A = []
    list_answers = []
    list_answer_gt_short = []
    list_answer_gt_coordinates_2d = []

    for sample_idx in tqdm(range(start_sample_idx, end_sample_idx), desc="Loading data from file"):
        file_path = f'./exp-data-scale/{dataset}/thoughts/{model}--{method}--{dataset}--{sample_idx}.json'
        (
            distance_matrix, num_chains, num_thoughts_each_chain, coordinates_2d, normed_A, answers, answer_gt_short, answer_gt_coordinates_2d
        ) = load_chain_data_and_plot(thoughts_file=file_path, tool=tool, plot=False)

        list_distance_matrix.append(distance_matrix)
        list_num_chains.append(num_chains)
        list_num_thoughts_each_chain.append(num_thoughts_each_chain)
        list_coordinates_2d.append(coordinates_2d)
        list_normed_A.append(normed_A)
        list_answers.append(answers)
        list_answer_gt_short.append(answer_gt_short)
        list_answer_gt_coordinates_2d.append(answer_gt_coordinates_2d)

    return list_distance_matrix, list_num_chains, list_num_thoughts_each_chain, list_coordinates_2d, list_answers, list_answer_gt_short, list_normed_A

'''
    Sample data --> Chain data
'''

def preprocess_data(
        list_distance_matrix, list_num_chains, 
        list_num_thoughts_each_chain, list_coordinates_2d, 
        list_answers, list_answer_gt_short,
        mode="cls"
    ):
    training_data_sample_2d = []
    training_data_sample_matrix = []
    for sample_idx in trange(len(list_answers), desc="Processing data"):
        num_chains = list_num_chains[sample_idx]
        num_thoughts_each_chain = list_num_thoughts_each_chain[sample_idx]
        coordinates_2d = list_coordinates_2d[sample_idx]
        distance_matrix = list_distance_matrix[sample_idx]
        model_answers = list_answers[sample_idx]
        gt_answer = list_answer_gt_short[sample_idx]

        training_data_chain_2d = []
        training_data_chain_matrix = []
        for chain_idx in range(num_chains):
            start_idx, end_idx = sum(num_thoughts_each_chain[:chain_idx]), sum(num_thoughts_each_chain[:chain_idx+1])
            chain_coordinates_2d = coordinates_2d[start_idx:end_idx, :]
            chain_matrix = distance_matrix[start_idx:end_idx, :]
            if mode == "cls":
                training_data_chain_2d.append((chain_coordinates_2d, gt_answer)) # for classification
                training_data_chain_matrix.append((chain_matrix, gt_answer))
            elif mode == "reg":
                chain_answer = model_answers[chain_idx] if model_answers[chain_idx] else "A" # default answer for the chain with empty values
                training_data_chain_2d.append((chain_coordinates_2d, gt_answer==chain_answer)) # for regression
                training_data_chain_matrix.append((chain_matrix, gt_answer==chain_answer)) # for regression
            else:
                raise NotImplementedError

        training_data_sample_2d.append(training_data_chain_2d)
        training_data_sample_matrix.append(training_data_chain_matrix)
    
    return training_data_sample_2d, training_data_sample_matrix

'''
    Data Verifier
'''
def collect_valid_coordinates_and_answers(training_data_sample):
    """
    收集并处理训练数据中的有效坐标和答案
    处理形如(n, 5)的坐标数组，其中n可能不同
    """
    all_coords = []
    all_answers = []
    
    for chain_idx, chain in enumerate(training_data_sample):
        for item_idx, (coordinates, answer) in enumerate(chain):
            # 确保coordinates不为空
            if not len(coordinates):
                print(f"Empty coordinates at chain {chain_idx}, item {item_idx}")
                continue
                
            try:
                # 将坐标转换为float数组
                coords_array = np.array(coordinates, dtype=np.float64)
                
                # 检查数组是否包含有效数据
                if coords_array.size == 0 or np.any(np.isnan(coords_array)):
                    print(f"Invalid coordinates at chain {chain_idx}, item {item_idx}")
                    continue
                if coords_array.shape[1] == 6:
                    coords_array = coords_array[:, 1:] # 丢弃 第一列 T(X, Y)
                # 添加有效数据
                all_coords.append(coords_array)
                # 为每一行坐标添加对应的答案
                all_answers.extend([answer] * coords_array.shape[0])
                
            except (ValueError, TypeError) as e:
                print(f"Error processing coordinates at chain {chain_idx}, item {item_idx}: {e}")
                continue
    
    if not all_coords:
        print("No valid coordinates collected")
        return None, None
    
    try:
        # 垂直堆叠所有坐标数组
        combined_coords = np.vstack(all_coords)
        answers_array = np.array(all_answers)
        
        print(f"Combined coordinates shape: {combined_coords.shape}")
        print(f"Answers array shape: {answers_array.shape}")
        
        return combined_coords, answers_array
        
    except Exception as e:
        print(f"Error during final processing: {e}")
        return None, None

'''
    Training
'''
# TODO: add meta_information
def train_model(
        data: list = None,
        model_type: str = 'lgb',
        model_configs: dict = None,
        target_labels: list = None,  # 新增参数：指定标签列表
        verbose: bool = False,
    ):
    """
    统一的模型训练函数
    
    Args:
        data: 训练数据
        model_type: 模型类型 ('rf', 'lgb', 'xgb', 'cat')
        model_configs: 模型配置参数
        target_labels: 指定的标签列表，例如 ['A', 'B', 'C', 'D', 'E']
        verbose: 是否打印详细信息
    """
    # 默认配置
    default_configs = {
        'rf': {
            "n_estimators": 200,
            "max_depth": 10,
            "random_state": 42
        },
        'lgb': {
            "n_estimators": 200,
            "max_depth": 10,
            "random_state": 42,
            "learning_rate": 0.1
        },
        'xgb': {
            "n_estimators": 200,
            "max_depth": 10,
            "random_state": 42,
            "learning_rate": 0.1
        },
        'cat': {
            "iterations": 200,
            "depth": 10,
            "random_seed": 42,
            "learning_rate": 0.1,
            "verbose": False
        }
    }

    if model_configs is None:
        model_configs = default_configs[model_type]

    # 数据预处理
    X, y = collect_valid_coordinates_and_answers(data)
    X = np.array(X)
    y = np.array(y)

    # 获取实际的类别标签
    unique_labels = np.unique(y)
    n_classes = len(unique_labels)
    
    if target_labels is not None:
        if not set(unique_labels).issubset(set(target_labels)):
            unknown_labels = set(unique_labels) - set(target_labels)
            raise ValueError(f"Data contains unknown labels: {unknown_labels}")
        
        # 使用指定的标签顺序
        label_encoder = LabelEncoder()
        label_encoder.fit(target_labels)
        target_names = target_labels
    else:
        # 如果没有指定标签，使用数据中的标签
        label_encoder = LabelEncoder()
        label_encoder.fit(unique_labels)
        target_names = list(label_encoder.classes_)
    
    if verbose:
        print(f"Using target labels: {target_names}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 标签编码
    y_scaled = label_encoder.transform(y)
    
    # 记录原始标签到编码标签的映射
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    
    if verbose:
        print("Label mapping:", label_mapping)

    # Split data
    random_state = model_configs.get("random_state", 42)
    if model_type == 'cat':
        random_state = model_configs.get("random_seed", 42)
        
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=random_state,
        stratify=y_scaled
    )

    # 初始化分类器
    if model_type == 'rf':
        clf = RandomForestClassifier(
            n_estimators=model_configs.get("n_estimators", 200),
            max_depth=model_configs.get("max_depth", 10),
            random_state=model_configs.get("random_state", 42),
        )
    elif model_type == 'lgb':
        clf = LGBMClassifier(
            boosting_type='gbdt', 
            num_leaves=55, 
            reg_alpha=0.0, 
            reg_lambda=1,
            max_depth=15, 
            n_estimators=6000, 
            objective='multiclass',  # 修改为multiclass
            num_class=len(target_labels),  # 添加类别数量参数
            subsample=0.8, 
            colsample_bytree=0.8, 
            subsample_freq=1,
            learning_rate=0.06, 
            min_child_weight=1, 
            random_state=42, 
            n_jobs=4,
            verbose=-1,
            min_split_gain=0
        )
    elif model_type == 'xgb':
        clf = XGBClassifier(
            n_estimators=model_configs.get("n_estimators", 200),
            max_depth=model_configs.get("max_depth", 10),
            random_state=model_configs.get("random_state", 42),
            learning_rate=model_configs.get("learning_rate", 0.1)
        )
    elif model_type == 'cat':
        clf = CatBoostClassifier(
            iterations=model_configs.get("iterations", 200),
            depth=model_configs.get("depth", 10),
            random_seed=model_configs.get("random_seed", 42),
            learning_rate=model_configs.get("learning_rate", 0.1),
            verbose=model_configs.get("verbose", False)
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Train model
    print(f"==>Training {model_type.upper()}...")
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # 评估
    try:
        cv_scores = cross_val_score(clf, X_scaled, y_scaled, cv=5)
    except Exception as e:
        print(f"Warning: Cross-validation failed: {e}")
        cv_scores = np.array([])

    accuracy = (y_pred == y_test).mean()
    
    try:
        # 对预测结果进行逆变换以获得原始标签
        y_test_original = label_encoder.inverse_transform(y_test)
        y_pred_original = label_encoder.inverse_transform(y_pred)
        
        classification_rep = classification_report(
            y_test_original, y_pred_original,
            target_names=target_names,
            zero_division=0
        )
    except Exception as e:
        print(f"Warning: Error in generating classification report: {e}")
        classification_rep = "Could not generate classification report"

    if verbose:
        if len(cv_scores) > 0:
            print(f"Cross-validation scores: {cv_scores}")
            print(f"Average CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"Test Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_rep)
        print("\nActual class distribution:")
        unique_counts = np.unique(y, return_counts=True)
        for label, count in zip(*unique_counts):
            print(f"Class {label}: {count} samples")

    return {
        'model': clf,
        'accuracy': accuracy,
        'classification_report': classification_rep,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'cv_scores': cv_scores,
        'unique_labels': unique_labels,
        'n_classes': n_classes,
        'label_mapping': label_mapping,
        'target_names': target_names
    }

'''
    Evaluation
'''
# LLM original Acc
def vote_accuracy(list_answers, list_answer_gt_short):
    voted_answers = [Counter(row).most_common(1)[0][0] if row else '' 
                    for row in list_answers]
    return sum(v == g for v, g in zip(voted_answers, list_answer_gt_short)) / len(list_answer_gt_short)

def row_wise_accuracy(list_answers, list_answer_gt_short):
    row_accs = [sum(ans == gt for ans in row if ans != '') / len([a for a in row if a != ''])
                if any(a != '' for a in row) else 0.0
                for row, gt in zip(list_answers, list_answer_gt_short)]
    return sum(row_accs) / len(row_accs)


def load_chain_data(
        sample_idx,
        list_num_chains,
        list_num_thoughts_each_chain,
        list_coordinates_2d,
        list_distance_matrix,
        list_answers,
        list_answer_gt_short,
        mode="cls"
    ):
    num_chains = list_num_chains[sample_idx]
    num_thoughts_each_chain = list_num_thoughts_each_chain[sample_idx]
    coordinates_2d = list_coordinates_2d[sample_idx]
    distance_matrix = list_distance_matrix[sample_idx]
    model_answers = list_answers[sample_idx]
    gt_answer = list_answer_gt_short[sample_idx]

    training_data_chain_2d = []
    training_data_chain_matrix = []
    for chain_idx in range(num_chains):
        start_idx, end_idx = sum(num_thoughts_each_chain[:chain_idx]), sum(num_thoughts_each_chain[:chain_idx+1])
        chain_coordinates_2d = coordinates_2d[start_idx:end_idx, :]
        chain_matrix = distance_matrix[start_idx:end_idx, :]
        if mode == "cls":
            training_data_chain_2d.append((chain_coordinates_2d, gt_answer)) # for classification
            training_data_chain_matrix.append((chain_matrix, gt_answer))
        elif mode == "reg":
            chain_answer = model_answers[chain_idx] if model_answers[chain_idx] else "A" # default answer for the chain with empty values
            training_data_chain_2d.append((chain_coordinates_2d, gt_answer==chain_answer)) # for regression
            training_data_chain_matrix.append((chain_matrix, gt_answer==chain_answer)) # for regression
        else:
            raise NotImplementedError
        
    return training_data_chain_2d, training_data_chain_matrix

def chain_collect_valid_coords_and_answers(list_chain_coord_answers):
    all_coords = []
    all_answers = []
    for chain_idx, (coordinates, answer) in enumerate(list_chain_coord_answers):
        # 确保coordinates不为空
        if not len(coordinates):
            print(f"Empty coordinates at chain {chain_idx}")
            continue
            
        try:
            # 将坐标转换为float数组
            coords_array = np.array(coordinates, dtype=np.float64)
            
            # 检查数组是否包含有效数据
            if coords_array.size == 0 or np.any(np.isnan(coords_array)):
                print(f"Invalid coordinates at chain {chain_idx}")
                continue
            if coords_array.shape[1] == 6:
                coords_array = coords_array[:, 1:] # 丢弃 第一列 T(X, Y)
            # 添加有效数据
            all_coords.append(coords_array)
            # 为每一行坐标添加对应的答案
            all_answers.extend([answer] * coords_array.shape[0])
            
        except (ValueError, TypeError) as e:
            print(f"Error processing coordinates at chain {chain_idx}: {e}")
            continue

    if not all_coords:
        print("No valid coordinates collected")
        return None, None

    try:
        # 垂直堆叠所有坐标数组
        combined_coords = np.vstack(all_coords)
        answers_array = np.array(all_answers)
        
        return combined_coords, answers_array
        
    except Exception as e:
        print(f"Error during final processing: {e}")
        return None, None

def calculate_accuracies(y_pred, y_true):
    # 计算准确率
    acc = accuracy_score(y_true, y_pred)
    
    # 计算预测标签的众数（投票结果）
    pred_counter = Counter(y_pred)
    vote_result = pred_counter.most_common(1)[0][0]  # 获取最常见的预测值
    
    # 投票准确率：众数是否等于真实标签
    vote_acc = 1 if vote_result == y_true[0] else 0
    
    # 平均准确率：正确预测的比例
    correct_predictions = np.sum(y_pred == y_true)
    avg_acc = correct_predictions / len(y_true)

    return vote_acc, avg_acc

# Random Forest
def eval_random_forest(
        data: list = None, 
        x_scaler: StandardScaler = None,
        y_scaler: StandardScaler = None,
        model: RandomForestClassifier = None,
        true_labels_available=True, 
    ):
    X, y = chain_collect_valid_coords_and_answers(data)
    X = np.array(X)

    if true_labels_available:
        y = np.array(y)
        y = y_scaler.transform(y)
    
    # Scale features using the same scaler
    X_scaled = x_scaler.transform(X)

    # print("==> Evaluating...")
    # Make predictions
    y_pred = model.predict(X_scaled)
    y_pred_proba = model.predict_proba(X_scaled)
    vote_acc, avg_acc = calculate_accuracies(y_pred, y)

    # Calculate confidence scores
    confidences = np.max(y_pred_proba, axis=1)
    results = {
        'voting_acc': vote_acc,
        'avg_acc': avg_acc,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'confidences': confidences,
        'y': y,
    }
    return results

def get_rf_predictions_with_confidence(clf, X):
    """
    获取随机森林的预测结果和置信度
    
    Parameters:
    clf: RandomForestClassifier
    X: 输入特征
    
    Returns:
    predictions: 预测类别
    confidences: 预测置信度
    probabilities: 所有类别的概率
    """
    # 获取预测概率
    probabilities = clf.predict_proba(X)
    
    # 获取预测类别
    predictions = clf.predict(X)
    
    # 获取置信度（最大概率值）
    confidences = np.max(probabilities, axis=1)
    
    return predictions, confidences, probabilities

def analyze_rf_predictions(clf, X, y_true=None):
    """
    分析随机森林预测结果
    
    Parameters:
    clf: RandomForestClassifier
    X: 输入特征
    y_true: 真实标签（可选）
    """
    predictions, confidences, probabilities = get_rf_predictions_with_confidence(clf, X)
    
    print("\nConfidence Analysis:")
    print(f"Mean confidence: {np.mean(confidences):.3f}")
    print(f"Min confidence: {np.min(confidences):.3f}")
    print(f"Max confidence: {np.max(confidences):.3f}")
    
    # 置信度分布统计
    conf_ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    print("\nConfidence Distribution:")
    for start, end in conf_ranges:
        count = np.sum((confidences >= start) & (confidences < end))
        print(f"{start:.1f}-{end:.1f}: {count} predictions ({count/len(confidences)*100:.1f}%)")
    
    if y_true is not None:
        # 计算准确率
        accuracy = np.mean(predictions == y_true)
        print(f"\nOverall Accuracy: {accuracy:.3f}")
        
        # 分析不同置信度范围的准确率
        print("\nAccuracy by Confidence Range:")
        for start, end in conf_ranges:
            mask = (confidences >= start) & (confidences < end)
            if np.sum(mask) > 0:
                range_acc = np.mean(predictions[mask] == y_true[mask])
                print(f"Confidence {start:.1f}-{end:.1f}: {range_acc:.3f}")


def analyze_confidence_accuracy_correlation(predictions, true_labels, confidences):
    """
    计算随机森林预测的准确性与置信度之间的相关性
    
    参数:
    predictions: np.array, 模型预测的标签
    true_labels: np.array, 真实标签
    confidences: np.array, 预测的置信度分数
    
    返回:
    dict: 包含相关性分析结果
    """
    import numpy as np
    from scipy import stats

    # 计算每个预测是否正确 (1表示正确，0表示错误)
    prediction_correctness = (predictions == true_labels).astype(int)
    
    # 计算Pearson相关系数
    pearson_corr, pearson_p = stats.pearsonr(confidences, prediction_correctness)
    
    # 计算Spearman相关系数
    spearman_corr, spearman_p = stats.spearmanr(confidences, prediction_correctness)
    
    # 计算不同置信度阈值下的准确率
    thresholds = np.arange(0.1, 1.0, 0.1)
    threshold_accuracies = []
    samples_above_threshold = []
    
    for threshold in thresholds:
        mask = confidences >= threshold
        if np.sum(mask) > 0:  # 确保有样本
            acc = np.mean(prediction_correctness[mask])
            threshold_accuracies.append(acc)
            samples_above_threshold.append(np.sum(mask))
        else:
            threshold_accuracies.append(np.nan)
            samples_above_threshold.append(0)
    
    return {
        'pearson_correlation': pearson_corr,
        'pearson_p_value': pearson_p,
        'spearman_correlation': spearman_corr,
        'spearman_p_value': spearman_p,
        'confidence_thresholds': thresholds,
        'threshold_accuracies': threshold_accuracies,
        'samples_above_threshold': samples_above_threshold
    }