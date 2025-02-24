import re
import textwrap
import time

import gym
# import igraph as ig
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup

REACT_TEMPLATE='''Here's some examples: {examples}
You must carefully follow these rules:
1. Your answer must begin with 'Thought', 'Action', and 'Observation', and you need to number your answers sequentially.
2. Every time you give an answer, you should give one Thought, one Action, and one Observation, without anything else, but must, contain all of them! Don't just give me one Thought or one Action!
3. Each Thought, Action, and Observation must be numbered consistently with the last number I provided, and must not repeat the numbering from the earlier prompts.
4. You must only provide one step at a time.
5. Your calculation after the Action must be enclosed in square brackets [] and should not include units.
6. Don't provide cross answers. For example, don't start with 'Action' where you should give a 'Thought'. You must start with 'Thought'.
7. Don't provide any explanations or apologies, just follow the format strictly.
8. Don't using symbols like '?' in calculating.
Your task is to solve this problem, think carefully: {question}'''

def clean_str(p):
    return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")

def replace_percentages(text):
    pattern = r'(\d+)%'
    
    def replace_match(match):
        number = int(match.group(1))
        return str(number / 100)

    result = re.sub(pattern, replace_match, text)
    return result

class textSpace(gym.spaces.Space):
    def contains(self, x) -> bool:
    
        return isinstance(x, str)

# ReAct environment
class Env(gym.Env):

    def __init__(self):
      
        super().__init__()
        self.page = None
        self.obs = None
        self.lookup_keyword = None
        self.lookup_list = None
        self.lookup_cnt = None
        self.steps = 0
        self.answer = None
        self.observation_space = self.action_space = textSpace()
        self.search_time = 0
        self.num_searches = 0
      
    def _get_obs(self):
        return self.obs

    def _get_info(self):
        return {"steps": self.steps, "answer": self.answer}

    def reset(self, seed=None, return_info=False, options=None):
      
        self.obs = ("Interact with Wikipedia using search[], lookup[], and "
                    "finish[].\n")
        self.page = None
        self.lookup_keyword = None
        self.lookup_list = None
        self.lookup_cnt = None
        self.steps = 0
        self.answer = None
        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation

    def construct_lookup_list(self, keyword):
      
        if self.page is None:
            return []
        paragraphs = self.page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        sentences = []
        for p in paragraphs:
            sentences += p.split('. ')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]

        parts = sentences
        parts = [p for p in parts if keyword.lower() in p.lower()]
        return parts

    @staticmethod
    def get_page_obs(page):
      
        paragraphs = page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        sentences = []
        for p in paragraphs:
            sentences += p.split('. ')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        return ' '.join(sentences[:5])

    def search_step(self, entity):
        entity_ = entity.replace(" ", "+")
        search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
        old_time = time.time()
        response_text = requests.get(search_url).text
        self.search_time += time.time() - old_time
        self.num_searches += 1
        soup = BeautifulSoup(response_text, features="html.parser")
        result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
        if result_divs:
            self.result_titles = [clean_str(div.get_text().strip()) for div in result_divs]
            self.obs = f"Could not find {entity}. Similar: {self.result_titles[:5]}."
        else:
            page = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
            if any("may refer to:" in p for p in page):
                self.search_step("[" + entity + "]")
            else:
                self.page = ""
                for p in page:
                    if len(p.split(" ")) > 2:
                        self.page += clean_str(p)
                        if not p.endswith("\n"):
                            self.page += "\n"
                self.obs = self.get_page_obs(self.page)
                self.lookup_keyword = self.lookup_list = self.lookup_cnt = None
    
    def step(self, action):
        reward = 0
        done = False
        action = action.strip()
        if self.answer is not None:
            done = True
            return self.obs, reward, done, self._get_info()
        
        if "finish" in action.lower() and "]" in action.lower():
            start = action.lower().index("[") + len("[")
            end = action.lower().index("]")
            answer = action[start:end].replace(",", "").replace("?", "0").replace("x", "*")
            if '%' in answer:
                answer = replace_percentages(answer)
            try:
                self.answer = eval(answer)
                done = True
                self.obs = f"The answer is {answer}\n"
            except Exception:
                self.obs = f"Ohh, maybe something wrong...\n"

        elif "calculate" in action.lower() and "]" in action.lower():
            start = action.lower().index("[") + len("[")
            end = action.lower().index("]")
            expression = action[start:end].replace(",", "").replace("x", "*")
            if not any(char.isdigit() for char in expression):
                self.obs = f"Ohh, there is no numbers in {expression}, I can only calculate numbers...\n"
            elif expression == ' ':
                self.obs = f"Ohh, there is nothing in {expression}\n"
            elif '?' in expression:
                self.obs = f"Ohh, there is a '?' in {expression}, I can not calculate '?', I should use numbers.\n"
            elif '_' in expression:
                self.obs = f"Ohh, there is a '_' in {expression}, I can not calculate '_', I should use numbers.\n"
            else:
                if '%' in expression:
                    expression = replace_percentages(expression)
                try:
                    result = eval(expression)
                    result = round(result, 5)
                    self.obs = f"The result is {result}\n"
                except Exception as e:
                    self.obs = f"Ohh, maybe something wrong in {expression}\n"

        elif "search[" in action.lower() and action.endswith("]"):
            start = action.lower().index("search[") + len("search[")
            entity = action[start:-1]
            self.search_step(entity)

        elif "lookup[" in action.lower() and action.endswith("]"):
            start = action.lower().index("lookup[") + len("lookup[")
            keyword = action[start:-1]
            if self.lookup_keyword != keyword:
                self.lookup_keyword = keyword
                self.lookup_list = self.construct_lookup_list(keyword)
                self.lookup_cnt = 0
            if self.lookup_cnt >= len(self.lookup_list):
                self.obs = "No more results.\n"
            else:
                self.obs = f"(Result {self.lookup_cnt + 1} / {len(self.lookup_list)}) " + self.lookup_list[self.lookup_cnt]
                self.lookup_cnt += 1

        else:
            self.obs = action

        self.steps += 1

        return self.obs, reward, done, self._get_info()

    
    def get_time_info(self):
      speed = self.search_time / self.num_searches if self.num_searches else 0
      return {
          "call_speed": speed,
          "call_time": self.search_time,
          "num_calls": self.num_searches,
      }


# def plot_tree(df):
#     # Function to insert line breaks into long strings
#     def insert_line_breaks(text, limit=50):
#         # Use textwrap to wrap text by the specified limit
#         wrapped_text = textwrap.fill(text, width=limit)
#         return wrapped_text.replace('\n', '<br>')

#     # Create an igraph Graph from the DataFrame
#     edges = df.dropna(subset=['parent']).apply(lambda row: (int(row['parent']), row['id']), axis=1).values.tolist()
#     g = ig.Graph(edges=edges, directed=True)

#     # Calculate a tree layout using igraph
#     layout = g.layout_reingold_tilford(root=[0])

#     # Extract positions from layout and invert y-coordinates
#     scaled_positions = np.array(layout.coords) * 1.5  # Scaling for better visualization
#     scaled_positions[:, 1] = -scaled_positions[:, 1] # Invert y-coordinates to place root at top


#     # Create edge traces
#     edge_traces = []
#     for edge in edges:
#         x_start, y_start = scaled_positions[edge[0]]
#         x_end, y_end = scaled_positions[edge[1]]
#         edge_traces.append(
#             go.Scatter(
#                 x=[x_start, x_end, None],
#                 y=[y_start, y_end, None],
#                 mode='lines',
#                 line=dict(width=2, color='black')
#             )
#         )

#     # Create node trace with hover information
#     x_pos, y_pos = zip(*scaled_positions)
#     node_trace = go.Scatter(
#         x=x_pos,
#         y=y_pos,
#         # text=df['label'],
#         mode='markers',
#         marker=dict(size=20, color=px.colors.qualitative.Plotly[2]),
#         hovertext=[
#             f"<b>Sequence:</b> {insert_line_breaks(seq)}<br><b>Value:</b> {val}<br>"
#             for seq, val in zip(df['sequence'], df['value'])
#         ],
#         hoverinfo='text',
#         textposition="top center"
#     )

#     # Combine traces
#     fig = go.Figure(data=edge_traces + [node_trace])

#     # Set layout properties
#     fig.update_layout(
#         showlegend=False,
#         xaxis=dict(showline=True, showgrid=True, zeroline=False),
#         yaxis=dict(showline=True, showgrid=True, zeroline=False),
#         margin=dict(t=0, b=0, l=0, r=0),
#         hovermode='closest',
#         width=800,  # Set the width of the figure
#         height=600  # Set the height of the figure
#     )

#     fig.show()

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