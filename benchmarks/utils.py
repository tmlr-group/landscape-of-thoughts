import re
from .gsm8k import Gsm8k
from .BBH import bbh
from .long_bench import LongBench


MULTIPLE_CHOICE_TASKS = [
    "temporal_sequences",
    "disambiguation_qa",
    "date_understanding",
    "tracking_shuffled_objects_three_objects",
    "penguins_in_a_table",
    "geometric_shapes",
    "snarks",
    "ruin_names",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_five_objects",
    "logical_deduction_three_objects",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "movie_recommendation",
    "salient_translation_error_detection",
    "reasoning_about_colored_objects",
]
FREE_FORM_TASKS = [
    "multistep_arithmetic_two",
    "navigate",
    "dyck_languages",
    "word_sorting",
    "sports_understanding",
    "boolean_expressions",
    "object_counting",
    "formal_fallacies",
    "causal_judgement",
    "web_of_lies",
]

LONGBENCH_TASKS = [
    "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
            "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
            "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"
]




def get_data_class(name):
    if name == "gsm8k":
        return Gsm8k()
    elif name=="BBH" or name=="bbh":
        return bbh()
    elif name in LONGBENCH_TASKS:
        return LongBench(dataset_name=name)
    else:
        raise NotImplementedError

def answer_cleansing(dataset_name, pred,mode="FREE_FORM_TASKS"):
    # adopted from https://github.com/kojima-takeshi188/zero_shot_cot/blob/main/utils.py        
    if dataset_name in ("aqua", "commonsensqa"):
        pred = re.findall(r'A|B|C|D|E', pred)
    elif dataset_name == "bigbench_date":
        pred = re.findall(r'A|B|C|D|E|F', pred)
    elif dataset_name=="BBH":
        ans_line = pred.split("answer is ")
        # Expect to see 'answer is'. If not return whole string
        if len(ans_line) == 1:
            pred=ans_line
        else:
            pred = ans_line[-1].strip()

        if mode == "multiple_choice":
            options = [
                "(A)",
                "(B)",
                "(C)",
                "(D)",
                "(E)",
                "(F)",
                "(G)",
                "(H)",
                "(I)",
                "(J)",
                "(K)",
                "(L)",
                "(M)",
                "(N)",
                "(O)",
                "(P)",
                "(Q)",
                "(R)",
                "(S)",
                "(T)",
                "(U)",
                "(V)",
                "(W)",
                "(X)",
                "(Y)",
                "(Z)",
            ]
            for option in options:
                if option in pred:
                    pred = option[1]
                    break
        elif mode == "free_form":
            if pred[-1] == ".":
                pred = pred[:-1]
            return pred
    elif dataset_name in ("object_tracking"):
        pred = re.findall(r'A|B|C', pred)
    elif dataset_name in ("gsm8k", "addsub", "multiarith", "svamp", "singleeq"):
        pred = pred.replace(",", "")
        pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]
    elif dataset_name in ("strategyqa", "coin_flip"):
        pred = pred.lower()
        pred = re.sub(r"\"|\'|\n|\.|\s|\:|\,"," ", pred)
        pred = pred.split(" ")
        pred = [i for i in pred if i in ("yes", "no")]
    elif dataset_name == "last_letters":
        pred = re.sub(r"\"|\'|\n|\.|\s","", pred)
        pred = [pred]
    elif dataset_name in LONGBENCH_TASKS:
        pred = [pred.strip()]
    else:
        raise ValueError("dataset is not properly defined ...")

    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = ""
    else:
        pred = pred[-1]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]

    return pred
