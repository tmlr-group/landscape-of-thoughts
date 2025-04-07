from ..models.get_response import *
import openai

def llm_verify(ans, real_ans, judge_model='gpt-4-1106-preview'):
    prompt = "Below you will see two paragraphs of text, the first paragraph will be a solution or answer (not necessarily correct) to a science question, and the second paragraph will be the standard answer to this question. Please judge whether the answer obtained from the first solution is consistent with the standard answer in the mathematical sense, and output '0' or '1' directly according to the judgment, without outputting any other information. If the answers are consistent, please output '1'; otherwise, as long as the answers do not match, or the answer is not explicitly stated in the first paragraph nor is the latex expression output, please output '0'; if the relationship between the first answer and the standard answer is ambiguous, please output '0'."
    qry = prompt + 'Paragraph 1:' + ans + '\n' + 'Paragraph 2:' + real_ans + '\nOutput:'
    lbl = ''
    cnt = 5
    while lbl == '' and cnt:
        out = ''
        try:
            chat_comp = openai.ChatCompletion.create(model=judge_model, messages=[{"role": "user", "content": qry}])
            out = chat_comp.choices[0].message.content[0]
        except Exception as e:
            print(f'Error:{e}\n')
        if out == '0' or out == '1':
            lbl = out
        else:
            cnt -= 1
    if not cnt:
        return 0
    return int(lbl)
