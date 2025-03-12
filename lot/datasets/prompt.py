"""
Dataset-specific prompts and patterns for different reasoning methods.

This module contains predefined prompts and patterns for different datasets and reasoning methods.
"""

# Dataset-specific prompts for different reasoning methods
DATASET_PROMPTS = {
    "strategyqa": {
        "cot": """Question: Did the 40th president of the United States forward lolcats to his friends?
Options: A. yes, B. no.
Answer: Let's think step by step. The 40th president of the United States was Ronald Reagan, who died in 2004. The first recorded use of the term lolcat occurred in 2006. Thus, the 40th president of the United States could not forward lolcats to his friends. So the answer is (B).

Question: Would an uninsured person be more likely than an insured person to decline a CT scan?
Options: A. yes, B. no.
Answer: Let's think step by step. CT scans without insurance can be very expensive, so an uninsured person is likely to cost much more than an insured person. Thus, an uninsured person is more likely to to decline a CT scan than an insured person. So the answer is (A).

Question: Were mollusks an ingredient in the color purple?
Options: A. yes, B. no.
Answer: Let's think step by step. The purple glands in ancient Tyre were made from the murex trunculus, which is a mollusk. Thus, mollusks were an ingredient in the color purple. So the answer is (A).

Refer to the solution process of the above example to provide a step-by-step solution to the following problem. Your final answer should be in the format of 'The answer is (chosen binary-choice option)'.
Question: {question}
Your final answer should be in the format of 'The answer is (chosen binary-choice option)'.
Answer: Let's think step-by-step.""",
        "zero-shot-cot": """Question: {question}
Your final answer should be in the format of 'The answer is (chosen binary-choice option)'.
Answer: Let's think step by step""",
        "l2m": """Question: Did the 40th president of the United States forward lolcats to his friends?
Options: A. yes, B. no.
Answer: To answer the question "Did the 40th president of the United States forward lolcats to his friends?", we need to know: "Who was the 40th president of the United States?", "Which year did the 40th president of the United States die in?", "In what year did the first lolcat appear?", "Is the year the first lolcat appear before or the same as the year the 40th president of the United States die?".

Q: Who was the 40th president of the United States?
A: The 40th president of the United States was Ronald Reagan. So the answer is Ronald Reagan.

Q: Which year did the 40th president of the United States die in?
A: Ronald Reagan died in 2004. So the answer is 2004.

Q: When did the first lolcat appear?
A: The first recorded use of the term lolcat occurred in 2006. So the answer is 2006.

Q: Is the year the first lolcat appear before or the same as the year the 40th president of the United States die?
A: The first recorded use of the term lolcat occurred in 2006. The 40th president of the United States died in 2004. 2006 is not before or the same as 2004. Thus, the year the first lolcat appear is not before or the same as the year the 40th president of the United States die. So the answer is no.

Q: Did the 40th president of the United States forward lolcats to his friends?
A: The year the first lolcat appear is later than the year the 40th president of the United States die. Thus, the 40th president of the United States could not forward lolcats to his friends. So the answer is (B).

Refer to the solution process of the above example to provide a step-by-step solution to the following problem. Your final answer should be in the format of 'The answer is (chosen binary-choice option)'.
Question: {question}
Answer: To answer the question, we need to know:
""",
        "mcts": """Question: {question}
Your final answer should be in the format of 'The answer is (chosen binary-choice option)'.
Answer: Let's think step by step""",
        "tot": """Question: {question}
Your final answer should be in the format of 'The answer is (chosen binary-choice option)'.
Answer: Let's think step by step""",
    },
    "commonsenseqa": {
        "cot": """You are a highly advanced large language model with specialized expertise in commonsense reasoning, including applying commonsense principles and knowledges to making assumptions about the nature and characteristics of physical objects, events, and situations, as well as inferring human intentions and behaviors. You would be given a commonsense question with multiple options. 

Question: When a person is breathing in a paper bag what are they trying to do? 
Options: A. warm air; B. continue to live; C. going to sleep; D. hyperventilation; E. stay alive 
Answer: Let's think step-by-step. Step 1: Understand the Context Breathing into a paper bag is often a response to anxiety or panic. It is used to help manage hyperventilation, where a person is breathing too quickly or too deeply. Step 2: Analyze the Options A. warm air: This does not relate to the primary reason for using a paper bag. B. continue to live: This is somewhat relevant but too vague; it doesn't specify the action's context. C. going to sleep: This is not applicable in this situation. D. hyperventilation: This directly relates to the act of breathing into a paper bag, as it is commonly used to control symptoms of hyperventilation. E. stay alive: Similar to B, this is general and not specifically tied to the action of using a paper bag. Step 3: Select the Best Option The option that best captures the intention behind breathing into a paper bag is D. hyperventilation, as it directly addresses the behavior associated with that action. Conclusion: The answer is D 

Question: Many singers might have to do this at some point? 
Options: A. sound beautiful; B. clear throats; C. warm up; D. use microphones; E. create music
Answer: Let's think step-by-step. Step 1: Understand the Context The question is about common actions that singers might need to perform during their singing careers. Step 2: Analyze the Options A. sound beautiful: While this is a goal, it's not something they actively do. B. clear throats: This is a common action, but it's more of a reaction to a specific issue rather than a regular practice. C. warm up: This is a standard practice for singers to prepare their voices before performing. D. use microphones: Many singers do use microphones, but not all situations require them, such as small venues. E. create music: This is an overarching goal but not a specific action they do at a certain point. Step 3: Select the Best Option The most relevant action that singers regularly perform to prepare for singing is C. warm up. Conclusion: The answer is C 

Question: Where do people often go to leave? 
Options: A. town; B. train station; C. conference; D. on vacation; E. apartment 
Answer: Let's think step-by-step. Step 1: Understand the Context The question asks about common places where people typically go when they are leaving a location, which suggests a focus on transit or departure points. Step 2: Analyze the Options A. town: While people might leave a town, it's not a specific departure point. B. train station: This is a common place where people go specifically to leave on a train. C. conference: People attend conferences, but it's not a typical place to leave from in the context of travel. D. on vacation: This describes an activity rather than a physical location people leave from. E. apartment: While people might leave an apartment, it's not a formal departure point for travel. Step 3: Select the Best Option The most appropriate answer is B. train station, as it directly refers to a place specifically designed for departures. Conclusion: The answer is B.

Refer to the solution process of the above example to provide a step-by-step solution to the following problem. Your final answer should be in the format of 'The answer is (chosen multiple-choice option)'.
Question: {question}
Answer: Let's think step-by-step.
""",
        "zero-shot-cot": """You are a highly advanced large language model with specialized expertise in commonsense reasoning, including applying commonsense principles and knowledges to making assumptions about the nature and characteristics of physical objects, events, and situations, as well as inferring human intentions and behaviors. You would be given a commonsense question with multiple options. You should choose the best option to answer the question.
Question: {question}
Your final answer should be in the format of 'The answer is (chosen multiple-choice option)'.
Answer: Let's think step-by-step.
""",
        "l2m": """
Question: Where would you buy jeans in a place with a large number of indoor merchants?  
Options: A. shopping mall; B. laundromat; C. hospital; D. clothing store; E. thrift store  
Answer: To answer the question " Where would you buy jeans in a place with a large number of indoor merchants?", we need to know: "What types of places typically have a large number of indoor merchants?", "Among those places, where can you specifically buy jeans?"

Q: What types of places typically have a large number of indoor merchants?
A: A shopping mall is designed to house numerous retailers indoors. A clothing store is a single retailer, not multiple merchants. A thrift store may be part of a larger complex but typically stands alone. Laundromat and hospital do not fit the description. 

Q: Among those places, where can you specifically buy jeans?
A: A shopping mall is a place where you can buy jeans. The answer is (A)

Refer to the solution process of the above example to provide a step-by-step solution to the following problem. Your final answer should be in the format of 'The answer is (chosen multiple-choice option)'.
Question: {question}
Answer: To answer the question, we need to know:""",
        "mcts": """You are a highly advanced large language model with specialized expertise in commonsense reasoning, including applying commonsense principles and knowledges to making assumptions about the nature and characteristics of physical objects, events, and situations, as well as inferring human intentions and behaviors. You would be given a commonsense question with multiple options. You should choose the best option to answer the question.
Question: {question}
Your final answer should be in the format of 'The answer is (chosen multiple-choice option)'.
Answer: Let's think step-by-step.
""",
        "tot": """You are a highly advanced large language model with specialized expertise in commonsense reasoning, including applying commonsense principles and knowledges to making assumptions about the nature and characteristics of physical objects, events, and situations, as well as inferring human intentions and behaviors. You would be given a commonsense question with multiple options. You should choose the best option to answer the question.
Question: {question}
Your final answer should be in the format of 'The answer is (chosen multiple-choice option)'.
Answer: Let's think step-by-step.
""",
    },
    "mmlu": {
        "cot": """The following are multiple choice questions (with answers) about high school physics.

Question: A microwave oven is connected to an outlet, 120 V, and draws a current of 2 amps. At what rate is energy being used by the microwave oven?
Options: (A) 10 W (B) 30 W (C) 60 W (D) 240 W
Answer: Let's think step by step. Rate of energy usage is known as power; in an dissipative electrical circuit, power is given by voltage times current. So in our case, the power is 120 V times 2 amps, or 240 W. The answer is (D).

Question: A point charge, Q = +1 mC, is fixed at the origin. How much work is required to move a charge, Q = +8 µC, from the point (0, 4 meters) to the point (3 meters, 0)?
Options: (A) 3.5 J (B) 6.0 J (C) 22.5 J (D) 40 J
Answer: Let's think step by step. To calculate the work required to move a charge from one location to another in a fixed electric field, it is enough to calculate the potential difference between the two locations. Here, the potential only depends on the distance between the charges; it's $k q_1 q_2 / r$, where $k$ is Coulomb's constant. Plugging in values $q_1 = $ 1 mC, $q_2 = 8 \mu$ C, gives the answer as 5.992 J, which rounds to 6 J. The answer is (B).

Question: Which of the following conditions will ensure that angular momentum is conserved? I. Conservation of linear momentum II. Zero net external force III. Zero net external torque
Options: (A) I and II only (B) I and III only (C) II and III only (D) III only
Answer: Let's think step by step. Torque is defined as the change in angular momentum; if there is zero external torque, angular momentum is conserved. The answer is (D).

Refer to the solution process of the above example to provide a step-by-step solution to the following problem. Your final answer should be in the format of 'The answer is (chosen multiple-choice option)'.

Question: {question}""",
        "zero-shot-cot": """Question: {question}
Your final answer should be in the format of 'The answer is (chosen multiple-choice option)'.
Answer: Let's think step by step.
""",
        "l2m": """Question: A spherical conductor carries a net charge. How is this charge distributed on the sphere?
Options: (A) The charge is evenly distributed on the surface. (B) The charge resides on the surface only; the distribution of charge on the surface depends on what other charged objects are near the sphere. (C) The charge moves continually within the sphere. (D) The charge is distributed uniformly throughout the sphere.
Answer: To answer the question "What is the new average height?", we need to know: "Where is the charge located on the sphere?", "How is the charge distributed?"

Q: Where is the charge located on the sphere?
A: For a spherical conductor, the charge resides on the surface. This is because charges in a conductor move freely, and they spread out on the surface to minimize repulsive forces.

Q: How is the charge distributed?
A: The charge resides on the surface only, but the distribution of charge on the surface can be influenced by nearby charged objects. The answer is (B)

Refer to the solution process of the above example to provide a solution to the following problem. Your final answer should be in the format of 'The answer is (chosen multiple-choice option)'.
Question: {question}
Answer: To answer the question, we need to know:""",
        "mcts": """Given a physics problem, your task is to answer the question step-by-step in a clear and specific manner.
Question: {question}
Your final answer should be in the format of 'The answer is (chosen multiple-choice option)'.
Answer: Let's think step by step.""",
        "tot": """Given a physics problem, your task is to answer the question step-by-step in a clear and specific manner.
Question: {question}
Your final answer should be in the format of 'The answer is (chosen multiple-choice option)'.
Answer: Let's think step by step.""",
    },
    "aqua": {
        "cot": """Question: A class of 35 students has an average height of 180 cm. Seven students whose average height is 120 cm, left the class and seven others whose average height is 140 cm, joined. Calculate the new average height of the students of the class (in cm) is? 
Options: (A) 204.6 cm, (B) 404.6 cm, (C) 224.6 cm, (D) 184.0 cm, (E) 256.6 cm.
Answer: Let's think step by step. The total height of students before seven students left is 180 * 35 = 6300 cm.The total height of students who joined is 140 * 7  = 980 cm. The new total height of students after seven students joined is 6300 - 840 + 980 = 6440 cm. The new average height is 6440 / 35 = 184 cm. The answer is D.

Question: How much is 70% of 40 is greater than 4/5 of 25? 
Options: (A) 22, (B) 67, (C) 88, (D) 12, (E) 8.
Answer: Let's think step by step. 70% of 40 is 40 * 0.7 = 28. 4/5 of 25 is 25 * 4/5 = 20. 70% of 40 is greater than 4/5 of 25 by 28 - 20 = 8. The answer is E.

Question: What is the average of the first 21 multiples of 7? 
Options: (A) 22, (B) 77, C) 88, (D) 21, (E) 65.
Answer: Let's think step by step. The sum of the first 21 multiples of 7 is 7 * (1+2+….+21). After simplification, 7 * ((21x22) / 2) = 1617. The average of the first 21 multiples of 7 is 1617 / 21 = 77. Therefore, among A through E, the answer is B.

Refer to the solution process of the above example to provide a step-by-step solution to the following problem. Your final answer should be in the format of 'The answer is (chosen multiple-choice option)
Q: {question}
Answer: Let's think step by step.
""",
        "zero-shot-cot": """Question: {question}
Your final answer should be in the format of 'The answer is (chosen multiple-choice option).
Answer: Let's think step by step. """,
        "l2m": """Question: A class of 35 students has an average height of 180 cm. Seven students whose average height is 120 cm, left the class and seven others whose average height is 140 cm, joined. Calculate the new average height of the students of the class (in cm) is? 
Options: A)204.6 cm, B)404.6 cm, C)224.6 cm, D)184.0 cm, E)256.6 cm
Answer: To answer the question "What is the new average height?", we need to know: "What is the total height of students before seven students left?", "What is the total height of students who left?", "What is the total height of students who joined?", "What is the new total height of students after seven students joined?". 

Q: What is the total height of students before seven students left?
A: Total height = 180 * 35 = 6300 cm. The answer is 6300 cm.

Q: What is the total height of students who joined?
A: Total height = 140 * 7  = 980 cm. The answer is 980 cm.

Q: What is the new total height of students after seven students joined?
A: Total height = 6300 - 840 + 980 = 6440 cm. The answer is 6440 cm.

Q: What is the new average height?
A: New average height is 6440 / 35 = 184 cm. The answer is D.

Refer to the solution process of the above example to provide a solution to the following problem.Your final answer should be in the format of "The answer is option (chosen multiple-choice option)"
Question: {question}
Answer: To answer the question, we need to know:""",
        "mcts": """Question: {question}
Your final answer should be in the format of 'The answer is (chosen multiple-choice option).
Answer: Let's think step by step. """,
        "tot": """Question: {question}
Your final answer should be in the format of 'The answer is (chosen multiple-choice option).
Answer: Let's think step by step. """,
    }
}

# Dataset-specific answer extraction patterns
DATASET_PATTERNS = {
    "mmlu": r'A|B|C|D',
    "strategyqa": r'A|B',
    "commonsenseqa": r'A|B|C|D|E',
    "aqua": r'A|B|C|D|E'
}

# Answer index mapper
ANSWER_IDX_MAPPER = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4
}

def get_prompt(dataset_name: str, method: str, default: str = None) -> str:
    """
    Get the prompt template for a specific dataset and reasoning method.
    
    Args:
        dataset_name (str): Name of the dataset.
        method (str): Reasoning method (cot, standard, tot, mcts).
        default (str, optional): Default prompt template if not found. Defaults to None.
        
    Returns:
        str: The prompt template, or default if not found.
    """
    dataset_name = dataset_name.lower()
    method = method.lower()
    
    if dataset_name in DATASET_PROMPTS and method in DATASET_PROMPTS[dataset_name]:
        return DATASET_PROMPTS[dataset_name][method]
    
    return default

def get_answer_pattern(dataset_name: str, default: str = r'A|B|C|D|E') -> str:
    """
    Get the answer extraction pattern for a specific dataset.
    
    Args:
        dataset_name (str): Name of the dataset.
        default (str, optional): Default pattern if not found. Defaults to r'A|B|C|D|E'.
        
    Returns:
        str: The answer extraction pattern, or default if not found.
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name in DATASET_PATTERNS:
        return DATASET_PATTERNS[dataset_name]
    
    return default

def get_answer_index(answer: str) -> int:
    """
    Get the index of an answer option.
    
    Args:
        answer (str): Answer option (A, B, C, D, E).
        
    Returns:
        int: The index of the answer option, or -1 if not found.
    """
    return ANSWER_IDX_MAPPER.get(answer, -1) 