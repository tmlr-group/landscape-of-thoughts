import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from fire import Fire
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from tqdm import tqdm

from utils.step_3_utils import *

'''
    Data utils
'''
def split_array(shapes, full_array):
    row_counts = [shape[0] for shape in shapes]
    assert sum(row_counts) == full_array.shape[0], "The total number of rows does not match"
    split_points = np.cumsum(row_counts)[:-1]
    return np.split(full_array, split_points)

def loading_data_from_file(model='Meta-Llama-3.1-70B-Instruct-Turbo', dataset='aqua', method="cot", total_sample=50, ROOT="./Landscape-Data"):
    # Load data
    ########################################
    plot_datas = {} 
    distance_matries = []
    num_all_thoughts_w_start_list = []

    for sample_idx in tqdm(range(total_sample), ncols=total_sample):
        # file_path = f'./exp-data-scale/{dataset}/thoughts/{model}--{method}--{dataset}--{sample_idx}.json'
        file_path = f'{ROOT}/{dataset}/thoughts/{model}--{method}--{dataset}--{sample_idx}.json'
        (distance_matrix, num_thoughts_each_chain, num_chains, num_all_thoughts, all_answers, answer_gt_short) = load_data(thoughts_file=file_path)
        
        plot_datas[sample_idx] = {
            "num_thoughts_each_chain": num_thoughts_each_chain,
            "num_chains": num_chains,
            "num_all_thoughts": num_all_thoughts,
            "all_answers": all_answers,
            "answer_gt_short": answer_gt_short
        }

        distance_matries.append(distance_matrix)
        num_all_thoughts_w_start_list.append(num_all_thoughts+1) # add one row from A matrix
    distance_matries = np.concatenate(distance_matries)
    return distance_matries, num_all_thoughts_w_start_list, plot_datas

def process_data(model='Meta-Llama-3.1-70B-Instruct-Turbo', dataset='aqua', method="cot", plot_type='method', total_sample=50, ROOT="./Landscape-Data", ):
    distance_matrix_shape = []
    list_distance_matrix = []
    list_num_all_thoughts_w_start_list = []
    list_plot_data = []

    if plot_type == "model":
        # assert method == 'cot', "model should be cot"
        # assert dataset == 'aqua', "dataset should be aqua"
        for model in ['Llama-3.2-1B-Instruct', 'Llama-3.2-3B-Instruct', 'Meta-Llama-3.1-8B-Instruct-Turbo', 'Meta-Llama-3.1-70B-Instruct-Turbo']:
            distance_matries, num_all_thoughts_w_start_list, plot_datas = loading_data_from_file(model=model, dataset=dataset, method=method, total_sample=total_sample, ROOT=ROOT)
            list_distance_matrix.append(distance_matries)
            list_plot_data.append(plot_datas)
            list_num_all_thoughts_w_start_list.append(num_all_thoughts_w_start_list)
            distance_matrix_shape.append(distance_matries.shape)
    
    elif plot_type == "dataset":
        # ! we cannot make all the sample with different num_answer to process together
        raise NotImplementedError
    
    elif plot_type == "method":
        # assert model == 'Meta-Llama-3.1-70B-Instruct-Turbo', "model should be 70B"
        # assert dataset == 'aqua', "dataset should be AQuA"
        for method in ['cot', 'l2m', 'mcts', 'tot']:
            distance_matries, num_all_thoughts_w_start_list, plot_datas = loading_data_from_file(model=model, dataset=dataset, method=method, total_sample=total_sample)
            list_distance_matrix.append(distance_matries)
            list_plot_data.append(plot_datas)
            list_num_all_thoughts_w_start_list.append(num_all_thoughts_w_start_list)
            distance_matrix_shape.append(distance_matries.shape)
    else:
        raise NotImplementedError

    fig_data = np.concatenate(list_distance_matrix)

    if dataset == "mmlu":
        target_A_matrix = np.ones((4,4)) * (1/4) 
    elif dataset == "strategyqa":
        target_A_matrix = np.ones((2,2)) * (1/3) 
    else:
        target_A_matrix = np.ones((5,5)) * (1/4) 
    target_A_matrix[np.diag_indices(target_A_matrix.shape[0])] = 0

    # concatenate all T and A(0-th row) (Nx(num_thoughts + 1), C), then concat the constant A matrix (C, C)
    tsne = TSNE(n_components=2, perplexity=10, random_state=42)
    all_T_constant_A_distance_matrix = tsne.fit_transform(np.concatenate([fig_data, target_A_matrix]))

    # split the Nx(num_thoughts + 1) back to sample-wise distance matrix
    if dataset == "mmlu":
        index = -4
    elif dataset == "strategyqa":
        index = -2
    else:
        index = -5
    all_T_2D, A_matrix_2D = all_T_constant_A_distance_matrix[:index, :], all_T_constant_A_distance_matrix[index:, :]
    list_all_T_2D = split_array(distance_matrix_shape, all_T_2D)

    return list_all_T_2D, A_matrix_2D, list_plot_data, list_num_all_thoughts_w_start_list


'''
    Plot utils
'''
def move_titles_to_bottom(fig, column_titles, y_position=-0.1, font_size=30):
    def update_annotation(a):
        if a.text in column_titles:
            a.update(y=y_position, font_size=font_size)
    fig.for_each_annotation(update_annotation)
    return fig

def draw(dataset_name, plot_datas, splited_T_2D, A_matrix_2D, num_all_thoughts_w_start_list):

    all_T_with_start_coordinate_matrix = split_list(num_all_thoughts_w_start_list, splited_T_2D)

    column_titles = [r'0-20% states', r'20-40% states', r'40-60% states', r'60-80% states', r'80-100% states']
    fig = make_subplots(rows=2, cols=5, 
                        vertical_spacing=0.01, horizontal_spacing=0.005,
                        column_titles=column_titles)

    # Collect points and separate them for correct/wrong chains
    wrong_chain_points = []
    correct_chain_points = []
    all_start_coordinates = []
    for sample_idx, plot_data in plot_datas.items():

        num_thoughts_each_chain, num_chains, _, all_answers, answer_gt_short = plot_data.values()
        try:
            temp_distance_matrix = all_T_with_start_coordinate_matrix[sample_idx]
        except:
            print(len(all_T_with_start_coordinate_matrix), sample_idx)
        thoughts_coordinates = np.array(temp_distance_matrix[:-1])
        start_coordinate = temp_distance_matrix[-1]
        all_start_coordinates.append(start_coordinate)
        
        # Collect points for each chain
        for chain_idx in range(num_chains):
            start_idx = sum(num_thoughts_each_chain[:chain_idx])
            end_idx = sum(num_thoughts_each_chain[:chain_idx+1])
            
            if end_idx <= start_idx:
                continue

            chain_points = thoughts_coordinates[start_idx:end_idx]

            if len(chain_points) <= 1:
                continue

            chain_data = {
                'points': chain_points,
                'start': start_coordinate
            }

            if all_answers[chain_idx] == answer_gt_short:
                correct_chain_points.append(chain_data)
            else:
                wrong_chain_points.append(chain_data)

    # Process both chains first
    wrong_x, wrong_y, wrong_weights, _, _ = process_chain_points(wrong_chain_points)
    correct_x, correct_y, correct_weights, _, _ = process_chain_points(correct_chain_points)

    # Calculate thresholds for both sets
    wrong_thresholds = np.percentile(wrong_weights, [20, 40, 60, 80])
    correct_thresholds = np.percentile(correct_weights, [20, 40, 60, 80])

    # Lists to store segment data
    wrong_segments = []
    correct_segments = []
    # Process wrong chains
    for i in range(5):
        if i == 0:
            wrong_mask = wrong_weights <= wrong_thresholds[0]
            correct_mask = correct_weights <= correct_thresholds[0]
        elif i == 4:
            wrong_mask = wrong_weights > wrong_thresholds[3]
            correct_mask = correct_weights > correct_thresholds[3]
        else:
            wrong_mask = (wrong_weights > wrong_thresholds[i-1]) & (wrong_weights <= wrong_thresholds[i])
            correct_mask = (correct_weights > correct_thresholds[i-1]) & (correct_weights <= correct_thresholds[i])

        # Get segments for both wrong and correct
        wrong_x_segment = np.array(wrong_x)[wrong_mask]
        wrong_y_segment = np.array(wrong_y)[wrong_mask]
        correct_x_segment = np.array(correct_x)[correct_mask]
        correct_y_segment = np.array(correct_y)[correct_mask]

        # Store segments and their scales
        wrong_segments.append((wrong_x_segment, wrong_y_segment))
        correct_segments.append((correct_x_segment, correct_y_segment))

    # Plot wrong chains (top subplot)
    #######################################
    for i in range(5):
        wrong_x_segment, wrong_y_segment = wrong_segments[i]
        correct_x_segment, correct_y_segment = correct_segments[i]

        fig.add_trace(
            go.Histogram2dContour(
                x=wrong_x_segment,
                y=wrong_y_segment,
                colorscale="Reds",
                showscale=False,
                histfunc='count',
                contours=dict(
                    showlines=True,
                    coloring='fill'
                ),
                autocontour=True,
                opacity=0.6,
                name=f'Wrong Range {i+1}'
            ),
            row=1, col=i+1
        )

        fig.add_trace(
            go.Histogram2dContour(
                x=correct_x_segment,
                y=correct_y_segment,
                colorscale="Blues",
                showscale=False,
                histfunc='count',
                contours=dict(
                    showlines=True,
                    coloring='fill'
                ),
                autocontour=True,
                opacity=0.6,
                name=f'Correct Range {i+1}'
            ),
            row=2, col=i+1
        )

        # fig.add_trace(go.Scatter(x=correct_x_segment,y=correct_y_segment,mode='markers',marker=dict(size=4,color='gray',colorscale='Greys',showscale=False,),showlegend=False,),row=2, col=i+1)


    # Add anchors to both plots
    if dataset_name == "mmlu":
        labels_anchors = ['A', 'B', 'C', 'D']
    elif dataset_name == "strategyqa":
        labels_anchors = ['A', 'B']
    else:
        labels_anchors = ['A', 'B', 'C', 'D', 'E']

    # Add anchors to both subplots
    for idx, anchor_name in enumerate(labels_anchors):
        if idx == 0:  # the first anchor is the correct one 
            marker_symbol = 'star'
            marker_color = "green"
        else: 
            marker_symbol = 'x'
            marker_color = "red"

        # Add to top subplot
        for col_idx in range(5):
            fig.add_trace(
                go.Scatter(
                    x=[A_matrix_2D[idx, 0]], 
                    y=[A_matrix_2D[idx, 1]], 
                    mode='markers',
                    marker=dict(
                        symbol=marker_symbol, 
                        size=18, 
                        line_width=0.5, 
                        color=marker_color,
                        opacity=0.8, # transparency
                    ),
                    showlegend=False,
                ),
                row=1, col=col_idx+1
            )

        # Add to bottom subplot
        for col_idx in range(5):
            fig.add_trace(
                go.Scatter(
                    x=[A_matrix_2D[idx, 0]], 
                    y=[A_matrix_2D[idx, 1]], 
                    mode='markers',
                    marker=dict(
                        symbol=marker_symbol, 
                        size=18, 
                        line_width=0.5, 
                        color=marker_color,
                        opacity=0.8, # transparency
                    ),
                    showlegend=False,
                ),
                row=2, col=col_idx+1
            )

    # move the subplot title to bottom
    fig = move_titles_to_bottom(fig, column_titles=column_titles, y_position=-0.12)

    # Update both subplots to remove axes and maintain same range
    for row in [1, 2]:
        for i in range(1, 6):
            fig.update_xaxes(
                row=row, 
                col=i,
                showticklabels=False,
                showgrid=True,           # Enable grid
                gridwidth=1,             # Grid line width
                gridcolor='lightgray',   # Grid line color
                zeroline=False,          # Hide zero line
                showline=False,          # Show axis line
                linewidth=1,             # Axis line width
                linecolor='black',       # Axis line color
                mirror=True,             # Mirror axis line
            )
            fig.update_yaxes(
                row=row, 
                col=i,
                showticklabels=False,
                showgrid=True,           # Enable grid
                gridwidth=1,             # Grid line width
                gridcolor='lightgray',   # Grid line color
                zeroline=False,          # Hide zero line
                showline=False,          # Show axis line
                linewidth=1,             # Axis line width
                linecolor='black',       # Axis line color
                mirror=True,             # Mirror axis line
            )

    # Update layout for a tighter plot
    fig.update_layout(
        height=350,  # Reduced height
        width=1500,
        margin=dict(l=5, r=5, t=5, b=37),  # Remove margins
        plot_bgcolor='white',  # Set plot background to white
        paper_bgcolor='white', # Set paper background to white
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
                gridcolor="black",
                zerolinecolor="black",
            ),
            
            yaxis=dict(
                title_font_color="black", 
                tickfont_color="black",
                linecolor="black",
                gridcolor="black",
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

def main(method="", model_name="", dataset_name="", ROOT="./Landscape-Data",):

    METHODS = [method] if method else ['cot', 'l2m', 'mcts', 'tot']
    MODELS = [model_name] if model_name else ['Llama-3.2-1B-Instruct', 'Llama-3.2-3B-Instruct', 'Meta-Llama-3.1-8B-Instruct-Turbo', 'Meta-Llama-3.1-70B-Instruct-Turbo']
    DATASETS = [dataset_name] if dataset_name else ['aqua', 'mmlu', 'commonsenseqa', 'strategyqa']

    for model in MODELS:
        for dataset in DATASETS: 
            list_all_T_2D, A_matrix_2D, list_plot_data, list_num_all_thoughts_w_start_list = process_data(
                model=model, 
                dataset=dataset, 
                plot_type='method',
                total_sample=50,
                ROOT=ROOT
            )
            method_idx = 0
            for plot_datas, splited_T_2D, num_all_thoughts_w_start_list in zip(list_plot_data, list_all_T_2D, list_num_all_thoughts_w_start_list):
                save_path = f"figures/landscape/FIG1_{model}-{dataset}-{METHODS[method_idx]}.png"
                fig = draw(
                    dataset_name=dataset, 
                    plot_datas=plot_datas, splited_T_2D=splited_T_2D, A_matrix_2D=A_matrix_2D, num_all_thoughts_w_start_list=num_all_thoughts_w_start_list, 
                )
                if not method: # if not specific method
                    method_idx += 1
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                print(f"==> save figure to:{save_path}")
                pio.write_image(fig, save_path, scale=6, width=1500, height=350)

if __name__ == "__main__":
    Fire(main)