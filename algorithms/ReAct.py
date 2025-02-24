import requests
from .utils import Env, REACT_TEMPLATE
from .base import Algorithm

class ReAct(Algorithm):
    """
    ReAct prompt:
    including ReAct
    """
    def __init__(self, method='react'):
        self.MAX_ATTEMPTS = 10
        self.method = method
        self.prompt_name = method + '_prompt'
        self.env = Env()

    def step(self, action):
        attempts = 0
        while attempts < self.MAX_ATTEMPTS:
            try:
                return self.env.step(action)
            except requests.exceptions.Timeout:
                attempts += 1

    def do_reasoning(self, question, rounds=10):
        thoughts = []
        actions = []
        observations = []
        step_strs = []

        inputs = REACT_TEMPLATE.format(examples="\n".join(self.examples), question=self.question_template.format(question=question))

        n_calls, n_badcalls = 0, 0
        done = False
        
        for i in range(1, rounds):
            thought_action = self.model.generate(
                inputs+f"Thought {i}:", 
                stop=[f"\nObservation ", "assistant"]
            ).text[0]
            
            n_calls += 1
            try:
                thought, action = thought_action.strip().split(f"\nAction {i}: ")
            except ValueError:
                try:
                    action = thought_action.strip().split("Action ")[1]
                    action = action[3:]
                    thought = "I need to " + action
                except IndexError:
                    try:
                        thought = thought_action.strip().split("\nAction ")[1].split("Thought ")[-1]
                        action = "I need to do something..."
                    except:
                        thought = thought_action.strip().split('\n')[0]
                        action = self.model.generate(inputs+f"Thought {i}:", stop=[f"\n"]).text[0].strip()
            
            obs, r, done, info = self.step(action)
            obs = obs.replace('\\n', '')
            if "Thought" in thought:
                step_str = f"{thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
            else:
                step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
            step_strs.append(step_str)
            inputs += step_str
            
            if done:
                break
        
        if not done:
            obs, r, done, info = self.step("finish[0]")
            
        result = "".join(step_strs)
        
        return result
