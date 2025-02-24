import os

import numpy as np
import plotly.io as pio
from joblib import dump, load

from utils.visual_utils import get_sample_distance_matrix


def split_array_by_weight_percentiles(data, n_segments=5):
    # Calculate percentile thresholds
    percentiles = np.linspace(0, 100, n_segments + 1)
    thresholds = np.percentile(data[:, 0], percentiles)
    
    segments = []
    for i in range(n_segments):
        if i == 0:
            mask = data[:, 0] <= thresholds[1]
        elif i == n_segments - 1:
            mask = data[:, 0] > thresholds[-2]
        else:
            mask = (data[:, 0] > thresholds[i]) & (data[:, 0] <= thresholds[i + 1])
        
        segments.append(data[mask])
    return segments


def get_consistency_data(model, dataset, method, root='exp-data-scale_full'):
    datas = get_sample_distance_matrix(model=model, dataset=dataset, method=method, root=root)
    all_sample_chains = []
    all_sample_answers = []
    all_sample_gts = []
    for _, plot_data in datas.items():
        distance_matrix, coordinates_2d, num_thoughts_each_chain, num_chains, all_answers, answer_gt_short = plot_data.values()
        # Collect points for each chain
        sample_chains = []
        for chain_idx in range(num_chains):
            start_idx = sum(num_thoughts_each_chain[:chain_idx])
            end_idx = sum(num_thoughts_each_chain[:chain_idx+1])
            if end_idx <= start_idx:
                continue
            if dataset == "strategyqa":
                if distance_matrix[start_idx:end_idx, :].shape[1] == 3:
                    chain_distance = distance_matrix[start_idx:end_idx, 1:] # remove the (s, T)
                else:
                    chain_distance = distance_matrix[start_idx:end_idx, :]
            elif dataset == "mmlu":
                if distance_matrix[start_idx:end_idx, :].shape[1] == 5:
                    chain_distance = distance_matrix[start_idx:end_idx, 1:] # remove the (s, T)
                else:
                    chain_distance = distance_matrix[start_idx:end_idx, :]
            else:
                if distance_matrix[start_idx:end_idx, :].shape[1] == 6:
                    chain_distance = distance_matrix[start_idx:end_idx, 1:] # remove the (s, T)
                else:
                    chain_distance = distance_matrix[start_idx:end_idx, :]
            
            sample_chains.append(chain_distance)
        all_sample_answers.append(all_answers)
        all_sample_gts.append(answer_gt_short)
        all_sample_chains.append(sample_chains)
    
    return all_sample_chains, all_sample_answers, all_sample_gts

def plot_chunk_accuracy(records, model, dataset, method, acc_calc_type='prediction'):
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Initialize lists for accuracies, separated by correctness
    max_chunk_idx = 5
    chunk_accs_correct = [[] for _ in range(max_chunk_idx + 1)]
    chunk_accs_incorrect = [[] for _ in range(max_chunk_idx + 1)]
    
    # Get chunk recorder
    chunk_recorder = records[model][dataset][method][acc_calc_type]
    
    # Collect accuracies
    for sample_idx in chunk_recorder:
        for chain_idx in chunk_recorder[sample_idx]:
            for chunk_idx in chunk_recorder[sample_idx][chain_idx]:
                try:
                    acc = chunk_recorder[sample_idx][chain_idx][chunk_idx]['acc']
                    correctness = chunk_recorder[sample_idx][chain_idx][chunk_idx]['correctness']
                    if acc is not None:
                        if correctness:
                            chunk_accs_correct[chunk_idx].append(acc)
                        else:
                            chunk_accs_incorrect[chunk_idx].append(acc)
                except:
                    continue
    
    # Calculate statistics for both groups
    valid_chunks_correct = []
    valid_chunks_incorrect = []
    mean_accs_correct = []
    std_errs_correct = []
    mean_accs_incorrect = []
    std_errs_incorrect = []
    
    for chunk_idx in range(max_chunk_idx + 1):
        accs_correct = chunk_accs_correct[chunk_idx]
        accs_incorrect = chunk_accs_incorrect[chunk_idx]
        
        if accs_correct:
            valid_chunks_correct.append(chunk_idx)
            mean_accs_correct.append(np.mean(accs_correct))
            std_errs_correct.append(np.std(accs_correct) / np.sqrt(len(accs_correct)))
            
        if accs_incorrect:
            valid_chunks_incorrect.append(chunk_idx)
            mean_accs_incorrect.append(np.mean(accs_incorrect))
            std_errs_incorrect.append(np.std(accs_incorrect) / np.sqrt(len(accs_incorrect)))
    
    # Create percentage labels for x-axis
    x_labels = [r'20%', r'40%', r'60%', r'80%', r'100%']
    
    # Create subplot structure
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.1)

    # Add bar chart for incorrect predictions to bottom subplot
    fig.add_trace(
        go.Bar(
            x=x_labels[:len(valid_chunks_incorrect)],
            y=mean_accs_incorrect,
            error_y=dict(
                type='data',
                array=std_errs_incorrect,
                visible=True
            ),
            marker=dict(
                pattern=dict(
                    shape='/',
                    solidity=0.7,
                    size=10,
                    fgcolor='white',
                    bgcolor='#EA8379'
                ),
                color='#EA8379'# Blue
            ),
        ),
        row=1, col=1
    )
    # Add bar chart for correct predictions to top subplot
    fig.add_trace(
        go.Bar(
            x=x_labels[:len(valid_chunks_correct)],
            y=mean_accs_correct,
            error_y=dict(
                type='data',
                array=std_errs_correct,
                visible=True
            ),
            marker=dict(
                pattern=dict(
                    shape='/',
                    solidity=0.7,
                    size=10,
                    fgcolor='white',
                    bgcolor='#7DAEE0'
                ),
                color='#7DAEE0'# Blue
            ),
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        height=600,
        width=500,
        margin=dict(l=5, r=5, t=5, b=5),  # Remove margins
        template="simple_white",
        showlegend=False,
    )

    # Update top subplot x-axis - hide ticks and labels
    fig.update_xaxes(
        showticklabels=False,  # Hide tick labels
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        row=1, col=1
    )

    # Update bottom subplot x-axis - show ticks and labels
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        tickfont=dict(size=32),
        autorange=True,
        row=2, col=1
    )


    # Update axes
    for i in range(1, 3):
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            tickfont=dict(size=32),
            autorange=True,
            row=i, col=1
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            title_font=dict(size=20),
            tickfont=dict(size=32),
            range=[0, 1.0],
            dtick=0.2,
            row=i, col=1
        )
    
    template = dict(
        layout=dict(
            font_color="black",
            paper_bgcolor="white",
            plot_bgcolor="white",
            title_font_color="black",
            legend_font_color="black",
            
            xaxis=dict(
                title_font_color="black",
                tickfont_color="black",
                linecolor="black",
                gridcolor="lightgray",
                zerolinecolor="black",
            ),
            
            yaxis=dict(
                title_font_color="black", 
                tickfont_color="black",
                linecolor="black",
                gridcolor="lightgray",
                zerolinecolor="black",
            ),
            
            hoverlabel=dict(
                font_color="black",
                bgcolor="white"
            ),
            
            annotations=[dict(font_color="black")],
            shapes=[dict(line_color="black")],
            
            coloraxis=dict(
                colorbar_tickfont_color="black",
                colorbar_title_font_color="black"
            ),
        )
    )

    fig.update_layout(template=template)
    
    return fig


records = {}
for model in [
    'Llama-3.2-1B-Instruct',
    'Llama-3.2-3B-Instruct',
    'Meta-Llama-3.1-8B-Instruct-Turbo',
    'Meta-Llama-3.1-70B-Instruct-Turbo'
]:
    records[model] = {}
    for dataset in ['aqua', 'mmlu', 'commonsenseqa', 'strategyqa']: 
        records[model][dataset] = {}
        methods = ['cot', 'l2m', 'tot', 'mcts']
        for method in methods:
            consistency_pkl_path = f"correlation_plot_data/{model}-{dataset}-{method}.pkl"
            if os.path.exists(consistency_pkl_path):
                all_sample_chains, all_sample_answers, all_sample_gts = load(consistency_pkl_path)
            else:
                print(f"==> Processing {model}-{dataset}-{method}")
                if method in ['tot', 'mcts']:
                    all_sample_chains, all_sample_answers, all_sample_gts = get_consistency_data(model, dataset, method, root='exp-data-searching')
                else:
                    all_sample_chains, all_sample_answers, all_sample_gts = get_consistency_data(model, dataset, method, root='exp-data-scale_full')
                dump((all_sample_chains, all_sample_answers, all_sample_gts), consistency_pkl_path)
            
            records[model][dataset][method] = {}
            
            prediction_chunk_recorder = {}
            last_thought_chunk_recorder = {}

            ANSWER_LIST = np.array(["A", "B", "C", "D", "E"])
            sample_mean_accs = []
            sample_mean_confs = []
            for sample_idx, sample_chains in enumerate(all_sample_chains): # 50 samples
                prediction_chunk_recorder[sample_idx] = {}
                last_thought_chunk_recorder[sample_idx] = {}

                chain_answers = all_sample_answers[sample_idx]
                sample_gt = all_sample_gts[sample_idx]
                confidence_list = []
                acc_list = []
                for chain_idx, chain in enumerate(sample_chains): # 10 CoT per sample
                    prediction_chunk_recorder[sample_idx][chain_idx] = {}
                    last_thought_chunk_recorder[sample_idx][chain_idx] = {}
                    # print(chain.shape) # (num_thoughts, n_anchors)

                    # * consistency of all thought with model prediction
                    model_pred = chain_answers[chain_idx]
                    if model_pred:
                        list_thought_chunk = split_array_by_weight_percentiles(chain, n_segments=5)
                        for chunk_idx, thought_chunk in enumerate(list_thought_chunk):
                            prediction_chunk_recorder[sample_idx][chain_idx][chunk_idx] = {}

                            if not len(thought_chunk):
                                prediction_chunk_recorder[sample_idx][chain_idx][chunk_idx] = {
                                    'acc': None,
                                    'conf': None,
                                    'correctness': model_pred == sample_gt
                                }
                            else:
                                list_thought_consistency = []
                                list_thought_conf = []
                                for thought in thought_chunk:
                                    # thouhgt: (5, ) distance matrix
                                    thought_answer_idx = np.argmin(thought)
                                    consistency = ANSWER_LIST[thought_answer_idx] == model_pred
                                    list_thought_consistency.append(consistency)
                                    list_thought_conf.append(thought[thought_answer_idx])

                                prediction_chunk_recorder[sample_idx][chain_idx][chunk_idx] = {
                                    'acc': np.mean(list_thought_consistency),
                                    'conf': np.mean(list_thought_conf),
                                    'correctness': model_pred == sample_gt
                                }
                        
                        # * consistency of all thought with last thought
                        if len(chain) <= 1:
                            continue
                        else:
                            chain_without_last_thought, last_thought = chain[:-1], chain[-1]
                            last_thought_answer = ANSWER_LIST[np.argmin(last_thought)]

                            list_thought_chunk = split_array_by_weight_percentiles(chain_without_last_thought, n_segments=5)
                            for chunk_idx, thought_chunk in enumerate(list_thought_chunk):
                                last_thought_chunk_recorder[sample_idx][chain_idx][chunk_idx] = {}
                                if not len(thought_chunk):
                                    last_thought_chunk_recorder[sample_idx][chain_idx][chunk_idx] = {
                                        'acc': None,
                                        'conf': None,
                                        'correctness': model_pred == sample_gt
                                    }
                                else:
                                    # average
                                    list_thought_consistency = []
                                    list_thought_conf = []
                                    for thought in thought_chunk:
                                        # thouhgt: (5, ) distance matrix
                                        thought_answer_idx = np.argmin(thought)
                                        consistency = ANSWER_LIST[thought_answer_idx] == last_thought_answer
                                        list_thought_consistency.append(consistency)
                                        list_thought_conf.append(thought[thought_answer_idx])

                                    last_thought_chunk_recorder[sample_idx][chain_idx][chunk_idx] = {
                                        'acc': np.mean(list_thought_consistency),
                                        'conf': np.mean(list_thought_conf),
                                        'correctness': model_pred == sample_gt
                                    }

            records[model][dataset][method] = {
                'prediction': prediction_chunk_recorder,
                'last_thought': last_thought_chunk_recorder
            }

acc_calc_type = 'last_thought' # prediction, last_thought
os.makedirs("imgs_landscape_analysis/chunk_consistency", exist_ok=True)
for model in [
    'Llama-3.2-1B-Instruct',
    'Llama-3.2-3B-Instruct',
    'Meta-Llama-3.1-8B-Instruct-Turbo',
    'Meta-Llama-3.1-70B-Instruct-Turbo'
]:
    for dataset in ['aqua', 'mmlu', 'commonsenseqa', 'strategyqa']: 
        methods = ['cot', 'l2m', 'tot', 'mcts']
        for method in methods:
            fig = plot_chunk_accuracy(records, model, dataset, method, acc_calc_type=acc_calc_type) # last_thought, prediction
            pio.write_image(fig, f'imgs_landscape_analysis/chunk_consistency/{model}-{method}-{dataset}-{acc_calc_type}.png', scale=6)
            print('Saved in '+f'imgs_landscape_analysis/chunk_consistency/{model}-{method}-{dataset}-{acc_calc_type}.png')
