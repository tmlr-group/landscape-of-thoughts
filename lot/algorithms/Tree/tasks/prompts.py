cot_prompt = '''
给定一个理科题目，你的任务是逐步地解答这个问题，解答过程要清晰且具体。
解答的格式限定为:"分析:...\n解答步骤:...\n综上所述，..."
请你先分析题目用到的知识点，然后分步地完成解答，最后概述最终答案。
题目: '''

cot_prompt_en = '''
Given a science problem, your task is to answer the question step-by-step in a clear and specific manner.
The format of the solution is limited to: "Solution: ...\nSummary: The final answer is $...$"
Please complete the answer step-by-step, and finally outline the final answer.
Problem: '''

MATH_cot_prompt = '''
You are supposed to provide a solution to a given problem.\n\n
Problem:\n{query}\nSolution: Let's think step by step.\n
'''

MATH_summary_prompt = '''
Given a math problem and its corresponding solution, your task is to extract the final answer obtained in the solution.
You should summarize the answer using the format: "The final answer is $...$". Replace "..." with the answer obtained in the solution.
Problem: '''

summary_prompt = '''
你的任务是给定一个理科题目和已完成的解答步骤，按照规定格式用一句话概述最终答案。
下面是几个例子，请学习

输入:
给定问题: 求使数列${n^{1/n}}$(n=1、2、3...为正整数)达到最大的$n$的值。
已有步骤:
步骤1: 考虑求导数：我们可以将数列 $n^{1/n}$ 视为函数 $f(x) = x^{1/x}$，然后求函数的导数 $f'(x)$。通过求导数，我们可以找到函数的增减性，进而确定数列的最大值所对应的正整数 $n$ 值。
步骤2: 基于上一步的思路，对于函数 $f(x) = x^{1/x}$，我们可以取自然对数来简化求导过程，得到 $g(x)=\\ln(f(x)) = \\frac{1}{x}\\ln(x)$，再对g(x)求导数。
步骤3: 经过计算，我们得到 $g(x)$ 导数为 $$-\\frac{1}{x^2}\\ln(x) + \\frac{1}{x^2}$$
步骤4: 接着，我们可以分析导数值的正负性。这个导数在 $x < e$ 时是正数，而在 $x > e$ 时是负数。这意味着函数 $f(n)$ 在 $n < e$ 时是递增的，而在 $n > e$ 时是递减的。
步骤5: 因此，在正整数范围内，最大值将出现在 $n = 3$ 或 $n = 2$（因为 $e \\approx 2.71828$）。而 $f(3) = 3^{1/3}$， $f(2) = 2^{1/2}$，所以 $n = 3$时函数值最大。
输出:
综上所述，在正整数范围内，数列 ${n^{1/n}}$ 的最大项的值为 $3^{1/3}$，对应的 $n$ 值为 3。

输入:
给定问题: 求函数$f(x)=-\\frac{1}{2}*(x^2)+x$在R上的最大值
已有步骤:
步骤1: 求导数：我们可以求出函数$f(x)$的导数$f'(x)$，即$f'(x)=-x+1$。通过求导数，我们可以找到函数的增减性，进而确定函数在R上的最大值所对应的$x$值。
步骤2: 根据$f'(x)=-x+1$，我们可以得到$f'(x)>0$的条件是$x<1$，即函数在$(-\\infty,1)$上单调递增；$f'(x)<0$的条件是$x>1$，即函数在$(1,+\\infty)$上单调递减。因此，$f(x)$在$x=1$处取得极大值，也就是最大值。
步骤3: 令$x=1$，我们可以得到函数在R上的最大值为$f(1)=-\\frac{1}{2}+1=\\frac{1}{2}$。
输出:
综上所述，函数$f(x)$在R上的最大值为$\\frac{1}{2}$，在$x=1$处取到。

输入:
给定问题: 讨论 p 取何值时, 广义积分$\\int_0^{+\\infty} \\frac{x^p \\ln x}{(1+x^2)^2}dx$收敛。
已有步骤:
步骤1: 考虑积分收敛的充分必要条件：我们可以记$J=\\int_0^{+\\infty} \\frac{x^p \\ln x}{(1+x^2)^2}dx $, $J_1=\\int_0^{1} \\frac{x^p \\ln x}{(1+x^2)^2}dx $, $J_2=\\int_1^{+\\infty} \\frac{x^p \\ln x}{(1+x^2)^2}dx $, 则广义积分$J$收敛当且仅当$J_1, J_2$都收敛。
步骤2: 首先考虑 $J_1$，当$x \\rightarrow 0^+$时，$\\frac{x^p \\ln x}{(1+x^2)^2} \\sim x^p \\ln x$ ，所以 $J_1$ 收敛当且仅当 $p > -1$。
步骤3: 然后考虑 $J_2$，当$x \\rightarrow +\\infty$时，$\\frac{x^p \\ln x}{(1+x^2)^2} \\sim \\frac{\\ln x}{x^{4-p}}$，所以 $J_2$ 收敛当且仅当 $p < 3$。
输出:
综上所述，p 需要满足 $p > -1$ 以及 $p < 3$，我们最终得出当 $-1 < p < 3$ 时，广义积分 $J$ 收敛。

输入:
给定问题: 求函数$f(x)=x+1$与直线$x=0$，$x=1$和x轴围成的图形的面积。
已有步骤:
步骤1: 我们可以通过计算定积分 $\\int_0^1 f(x) dx$ 来得到函数 $f(x) = x + 1$ 与直线 $x=0$，$x=1$ 和 $x$ 轴围成的图形的面积。
步骤2: 具体而言，上述定积分计算过程为$\\int_0^1 f(x) dx=\\int_0^1 (x+1) dx=\\frac{1}{2} + 1 = \\frac{3}{2}$。
输出:
综上所述，函数$f(x)=x+1$与直线$x=0$，$x=1$和x轴围成的图形的面积为$\\frac{3}{2}$。

输入:
给定问题: 请解决下面的问题，答案应为'A'，'B'，'C'，'D'中的一个。函数$f(x)=x+1$与直线$x=0$，$x=1$和x轴围成的图形的面积是_。A:1，B:1.5，C:2，D:2.5。
已有步骤:
步骤1: 我们可以通过计算定积分 $\\int_0^1 f(x) dx$ 来得到函数 $f(x) = x + 1$ 与直线 $x=0$，$x=1$ 和 $x$ 轴围成的图形的面积。
步骤2: 计算上述定积分，得到$\\int_0^1 f(x) dx=\\int_0^1 (x+1) dx=\\frac{1}{2} + 1 = \\frac{3}{2}$，即$1.5$，因此选项B正确。
输出:
综上所述，答案是B。

下面是你要给出概述的题目，请根据已有步骤得到的结果，按规定格式"综上所述，..."输出最终答案。

给定问题: '''

evaluate_summary_prompt_old = '''
你的任务是给定一个理科题目和已有的解答步骤，按固定格式输出最终答案，输出需转化为整数或者两位小数。
下面是几个例子，请学习

输入:
给定问题: 求使数列${n^{1/n}}$(n=1、2、3...为正整数)达到最大的$n$的值。
已有步骤:
步骤1: 考虑求导数：我们可以将数列 $n^{1/n}$ 视为函数 $f(x) = x^{1/x}$，然后求函数的导数 $f'(x)$。通过求导数，我们可以找到函数的增减性，进而确定数列的最大值所对应的正整数 $n$ 值。
步骤2: 基于上一步的思路，对于函数 $f(x) = x^{1/x}$，我们可以取自然对数来简化求导过程，得到 $g(x)=\\ln(f(x)) = \\frac{1}{x}\\ln(x)$，再对g(x)求导数。
步骤3: 经过计算，我们得到 $g(x)$ 导数为 $$-\\frac{1}{x^2}\\ln(x) + \\frac{1}{x^2}$$
步骤4: 接着，我们可以分析导数值的正负性。这个导数在 $x < e$ 时是正数，而在 $x > e$ 时是负数。这意味着函数 $f(n)$ 在 $n < e$ 时是递增的，而在 $n > e$ 时是递减的。
步骤5: 因此，在正整数范围内，最大值将出现在 $n = 3$ 或 $n = 2$（因为 $e \\approx 2.71828$）。而 $f(3) = 3^{1/3}$， $f(2) = 2^{1/2}$，所以 $n = 3$时函数值最大。
输出:
综上所述，最终答案是:3

输入:
给定问题: 求函数$f(x)=-\\frac{1}{2}*(x^2)+x$在R上的最大值。
已有步骤:
步骤1: 求导数：我们可以求出函数$f(x)$的导数$f'(x)$，即$f'(x)=-x+1$。通过求导数，我们可以找到函数的增减性，进而确定函数在R上的最大值所对应的$x$值。
步骤2: 根据$f'(x)=-x+1$，我们可以得到$f'(x)>0$的条件是$x<1$，即函数在$(-\\infty,1)$上单调递增；$f'(x)<0$的条件是$x>1$，即函数在$(1,+\\infty)$上单调递减。因此，$f(x)$在$x=1$处取得极大值，也就是最大值。
步骤3: 令$x=1$，我们可以得到函数在R上的最大值为$f(1)=-\\frac{1}{2}+1=\\frac{1}{2}$。
输出:
综上所述，最终答案是:0.50

输入:
给定问题: 如果广义积分$\\int_0^{+\\infty} \\frac{x^p \\ln x}{(1+x^2)^2}dx$收敛，那么整数 p 最大可以取多少？
已有步骤:
步骤1: 考虑积分收敛的充分必要条件：我们可以记$J=\\int_0^{+\\infty} \\frac{x^p \\ln x}{(1+x^2)^2}dx $, $J_1=\\int_0^{1} \\frac{x^p \\ln x}{(1+x^2)^2}dx $, $J_2=\\int_1^{+\\infty} \\frac{x^p \\ln x}{(1+x^2)^2}dx $, 则广义积分$J$收敛当且仅当$J_1, J_2$都收敛。
步骤2: 首先考虑 $J_1$，当$x \\rightarrow 0^+$时，$\\frac{x^p \\ln x}{(1+x^2)^2} \\sim x^p \\ln x$ ，所以 $J_1$ 收敛当且仅当 $p > -1$。
步骤3: 然后考虑 $J_2$，当$x \\rightarrow +\\infty$时，$\\frac{x^p \\ln x}{(1+x^2)^2} \\sim \\frac{\\ln x}{x^{4-p}}$，所以 $J_2$ 收敛当且仅当 $p < 3$。
步骤4: 因此，p 需要满足$-1 < p < 3$，因为 p 为整数，因此 p 最大为2。
输出:
综上所述，最终答案是:2

输入:
给定问题: 求函数$f(x)=-\\frac{1}{2}*(x^2)+2*x-1$在R上的最大值。
已有步骤:
步骤1: 求导数：我们可以求出函数$f(x)$的导数$f'(x)$，即$f'(x)=-x+2$。通过求导数，我们可以找到函数的增减性，进而确定函数在R上的最大值所对应的$x$值。
步骤2: 函数的极大值应该在导数为0的点取到，令$f'(x)=0$，得到$x=2$，且二阶导数$f''(2)=-1<0$，因此$x=2$为函数的最大值点。
步骤3: 令$x=2$，我们可以得到函数在R上的最大值为$f(2)=-2+4-1=1$。
输出:
综上所述，最终答案是:1

输入:
给定问题: 求函数$f(x)=x+1$与直线$x=0$，$x=1$和x轴围成的图形的面积。
已有步骤:
步骤1: 我们可以通过计算定积分 $\\int_0^1 f(x) dx$ 来得到函数 $f(x) = x + 1$ 与直线 $x=0$，$x=1$ 和 $x$ 轴围成的图形的面积。
步骤2: 具体而言，上述定积分计算过程为$\\int_0^1 f(x) dx=\\int_0^1 (x+1) dx=\\frac{1}{2} + 1 = \\frac{3}{2}$。
输出:
综上所述，最终答案是:1.50

下面是你要给出概述的题目，请按规定格式"综上所述，最终答案是:..."输出最终答案(对非整数答案均保留2位小数)。

给定问题: '''

evaluate_summary_prompt = '''
你的任务是给定一个理科题目和已有的解答步骤，按固定格式输出最终答案，输出需转化为整数或者两位小数。
下面是几个例子，请学习

输入:
给定问题: 求使数列${n^{1/n}}$(n=1、2、3...为正整数)达到最大的$n$的值。
已有步骤:
步骤1: 考虑求导数：我们可以将数列 $n^{1/n}$ 视为函数 $f(x) = x^{1/x}$，然后求函数的导数 $f'(x)$。通过求导数，我们可以找到函数的增减性，进而确定数列的最大值所对应的正整数 $n$ 值。
步骤2: 基于上一步的思路，对于函数 $f(x) = x^{1/x}$，我们可以取自然对数来简化求导过程，得到 $g(x)=\\ln(f(x)) = \\frac{1}{x}\\ln(x)$，再对g(x)求导数。
步骤3: 经过计算，我们得到 $g(x)$ 导数为 $$-\\frac{1}{x^2}\\ln(x) + \\frac{1}{x^2}$$
步骤4: 接着，我们可以分析导数值的正负性。这个导数在 $x < e$ 时是正数，而在 $x > e$ 时是负数。这意味着函数 $f(n)$ 在 $n < e$ 时是递增的，而在 $n > e$ 时是递减的。
步骤5: 因此，在正整数范围内，最大值将出现在 $n = 3$ 或 $n = 2$（因为 $e \\approx 2.71828$）。而 $f(3) = 3^{1/3}$， $f(2) = 2^{1/2}$，所以 $n = 3$时函数值最大。
输出:
综上所述，最终答案是:3

输入:
给定问题: 求函数$f(x)=-\\frac{1}{2}*(x^2)+x$在R上的最大值。
已有步骤:
步骤1: 求导数：我们可以求出函数$f(x)$的导数$f'(x)$，即$f'(x)=-x+1$。通过求导数，我们可以找到函数的增减性，进而确定函数在R上的最大值所对应的$x$值。
步骤2: 根据$f'(x)=-x+1$，我们可以得到$f'(x)>0$的条件是$x<1$，即函数在$(-\\infty,1)$上单调递增；$f'(x)<0$的条件是$x>1$，即函数在$(1,+\\infty)$上单调递减。因此，$f(x)$在$x=1$处取得极大值，也就是最大值。
步骤3: 令$x=1$，我们可以得到函数在R上的最大值为$f(1)=-\\frac{1}{2}+1=\\frac{1}{2}$。
输出:
综上所述，最终答案是:0.50

输入:
给定问题: 求函数$f(x)=-\\frac{1}{2}*(x^2)+2*x-1$在R上的最大值。
已有步骤:
步骤1: 求导数：我们可以求出函数$f(x)$的导数$f'(x)$，即$f'(x)=-x+2$。通过求导数，我们可以找到函数的增减性，进而确定函数在R上的最大值所对应的$x$值。
步骤2: 函数的极大值应该在导数为0的点取到，令$f'(x)=0$，得到$x=2$，且二阶导数$f''(2)=-1<0$，因此$x=2$为函数的最大值点。
步骤3: 令$x=2$，我们可以得到函数在R上的最大值为$f(2)=-2+4-1=1$。
输出:
综上所述，最终答案是:1

输入:
给定问题: 求函数$f(x)=x+1$与直线$x=0$，$x=1$和x轴围成的图形的面积。
已有步骤:
步骤1: 我们可以通过计算定积分 $\\int_0^1 f(x) dx$ 来得到函数 $f(x) = x + 1$ 与直线 $x=0$，$x=1$ 和 $x$ 轴围成的图形的面积。
步骤2: 具体而言，上述定积分计算过程为$\\int_0^1 f(x) dx=\\int_0^1 (x+1) dx=\\frac{1}{2} + 1 = \\frac{3}{2}$。
输出:
综上所述，最终答案是:1.50

下面是你要给出概述的题目，请按规定格式"综上所述，最终答案是:..."输出最终答案(对非整数答案均保留2位小数)。

给定问题: '''

general_evaluate_summary_prompt = '''
你的任务是给定一个理科题目和已有的解答步骤，按题目中规定的格式输出最终答案。
下面是几个例子，请学习

输入:
给定问题: 请解决下面的问题，答案应为整数。求使数列${n^{1/n}}$(n=1、2、3...为正整数)达到最大的$n$的值。
已有步骤:
步骤1: 考虑求导数：我们可以将数列 $n^{1/n}$ 视为函数 $f(x) = x^{1/x}$，然后求函数的导数 $f'(x)$。通过求导数，我们可以找到函数的增减性，进而确定数列的最大值所对应的正整数 $n$ 值。
步骤2: 基于上一步的思路，对于函数 $f(x) = x^{1/x}$，我们可以取自然对数来简化求导过程，得到 $g(x)=\\ln(f(x)) = \\frac{1}{x}\\ln(x)$，再对g(x)求导数。
步骤3: 经过计算，我们得到 $g(x)$ 导数为 $$-\\frac{1}{x^2}\\ln(x) + \\frac{1}{x^2}$$
步骤4: 接着，我们可以分析导数值的正负性。这个导数在 $x < e$ 时是正数，而在 $x > e$ 时是负数。这意味着函数 $f(n)$ 在 $n < e$ 时是递增的，而在 $n > e$ 时是递减的。
步骤5: 因此，在正整数范围内，最大值将出现在 $n = 3$ 或 $n = 2$（因为 $e \\approx 2.71828$）。而 $f(3) = 3^{1/3}$， $f(2) = 2^{1/2}$，所以 $n = 3$时函数值最大。
输出:
综上所述，最终答案是:3

输入:
给定问题: 请解决下面的问题，答案应为'A'，'B'，'C'，'D'中的一个。函数$f(x)=x+1$与直线$x=0$，$x=1$和x轴围成的图形的面积是_。A:1，B:1.5，C:2，D:2.5。
已有步骤:
步骤1: 我们可以通过计算定积分 $\\int_0^1 f(x) dx$ 来得到函数 $f(x) = x + 1$ 与直线 $x=0$，$x=1$ 和 $x$ 轴围成的图形的面积。
步骤2: 计算上述定积分，得到$\\int_0^1 f(x) dx=\\int_0^1 (x+1) dx=\\frac{1}{2} + 1 = \\frac{3}{2}$，即$1.5$，因此选项B正确。
输出:
综上所述，最终答案是:B

输入:
给定问题: 请判断下面的命题是否为真，如果为真，输出"yes"，否则输出"no"。高锰酸钾KMnO4中的锰元素Mn为+7价。
已有步骤:
步骤1: 我们可以用钾元素K和氧元素O的化合价结合总化合价为0来计算Mn元素的化合价。
步骤2: K元素只有+1价，而O元素常见为-2价。设Mn元素化合价为x，则$1+x+4*(-2)=0$，解得$x=7$。因此Mn元素为+7价，命题为真。
输出:
综上所述，最终答案是:yes

下面是你要给出最终答案的题目，请结合题目要求，采用规定格式"综上所述，最终答案是:..."输出最终答案。

给定问题: '''

single_proposal_prompt = '''
你的任务是给定一个理科问题和已有的解答步骤（并不是完整的答案），给出正确的下一步。下面是几个例子，请学习。

样例1
题目: 讨论 p 取何值时, 广义积分$\\int_0^{+\\infty} \\frac{x^p \\ln x}{(1+x^2)^2}dx$收敛。
已有步骤:
步骤1: 考虑积分收敛的充分必要条件：我们可以记$J=\\int_0^{+\\infty} \\frac{x^p \\ln x}{(1+x^2)^2}dx $, $J_1=\\int_0^{1} \\frac{x^p \\ln x}{(1+x^2)^2}dx $, $J_2=\\int_1^{+\\infty} \\frac{x^p \\ln x}{(1+x^2)^2}dx $, 则广义积分$J$收敛当且仅当$J_1, J_2$都收敛。

输出:
分析: 广义积分的收敛性是通过它的子区间上的积分来确定的。根据步骤1, 我们已经把原积分分解成了$J_1$ 和$J_2$ 两部分。为了确定$J$的收敛性，我们需要分别讨论这两部分。对于$J_1$, 在$[0,1]$范围内，当$x$接近0时会出现特定的行为，所以需要研究当$x$接近0时的表现。而对于$J_2$, 主要关注的是$x$接近$+\\infty$时的行为。
下一步: 为了分析$J_1$的收敛性，比较其与在$x \\to 0^+$时的已知函数（如$x^a \\ln x$其中$a > -1$时是收敛的）。具体地，我们可以选择适当的$q$，使得当$p > q$时，$\\frac{x^p \\ln x}{(1+x^2)^2} > x^q \\ln x$，从而推出$J_1$的收敛性；

样例2
题目: 讨论 p 取何值时, 广义积分$\\int_0^{+\\infty} \\frac{x^p \\ln x}{(1+x^2)^2}dx$收敛。
已有步骤:
步骤1: 考虑积分收敛的充分必要条件：我们可以记$J=\\int_0^{+\\infty} \\frac{x^p \\ln x}{(1+x^2)^2}dx $, $J_1=\\int_0^{1} \\frac{x^p \\ln x}{(1+x^2)^2}dx $, $J_2=\\int_1^{+\\infty} \\frac{x^p \\ln x}{(1+x^2)^2}dx $, 则广义积分$J$收敛当且仅当$J_1, J_2$都收敛。
步骤2: 首先考虑 $J_1$，当$x \\rightarrow 0^+$时，$\\frac{x^p \\ln x}{(1+x^2)^2} \\sim x^p \\ln x$ ，所以 $J_1$ 收敛当且仅当 $p > -1$。

输出:
分析: 给定的问题是探讨广义积分$J$在何时收敛。基于已有步骤，我们已经将这个广义积分分解为两个部分$J_1$和$J_2$，并指出了$J$的收敛性依赖于这两部分的收敛性。进一步，步骤2讨论了$J_1$在$x \\rightarrow 0^+$的情况，并给出了其收敛性的条件$p > -1$。下一步应当集中于$J_2$的分析，确定$p$值的范围使得$J_2$也收敛。
下一步: 考虑$J_2$，当$x \\rightarrow +\\infty$时，分析$\\frac{x^p \\ln x}{(1+x^2)^2}$的行为，从而判断$J_2$的收敛性，并得到$p$的条件。

样例3
题目: 求数列${n^{1/n}}$(n=1、2、3...为正整数)的最大项的值。
已有步骤:
步骤1: 首先，我们可以将数列 $n^{1/n}$ 视为函数 $f(x) = x^{1/x}$，然后求函数的导数 $f'(x)$。通过求导数，我们可以找到函数的增减性，进而确定数列的最大值所对应的正整数 $n$ 值。

输出:
分析: 为了找到一个函数在某个区间上的最大值或最小值，常用的方法是利用导数找到该函数的极值点。导数为0的点可能是极值点。在这个问题中，我们已经转化为寻找函数$f(x) = x^{1/x}$的最大值。要找到该函数的最大值，我们需要先求导数，然后解导数为0的方程找到可能的极值点，最后利用二阶导或者利用导数的正负来确定这些点是最大值点还是最小值点。
下一步: 明确求出函数 $f'(x)$ 的表达式，并求解 $f'(x) = 0$ 的解。这些解是函数$f(x)$可能的极值点；

样例4
题目: 设 $A$, $B$, $C$ 为随机事件, 且 $A$ 与 $B$ 互不相容, $A$ 与 $C$ 互不相容, $B$ 与 $C$ 相互独立  $P(A)=P(B)=P(C)=\\frac{1}{3}$, 则 $P(B \\cup C \mid A \\cup B \\cup C)= ？$
已有步骤:
步骤1: 根据分析部分，我们可以得到以下结果：+由互不相容事件的性质，我们知道$P(A\\cap B)=P(A\\cap C)=0$，即事件A与事件B的交集、事件A与事件C的交集的概率为0.

输出:
分析: 这个问题考察了概率论中的条件概率和事件的独立性。我们已经得知了事件$A$与$B$以及事件$A$与$C$是互不相容的，这意味着这两对事件不可能同时发生。此外，已知事件$B$与$C$是相互独立的，也就是说它们之间没有关联。基于这些信息，我们需要计算在已知事件$A \\cup B \\cup C$发生的情况下，事件$B \\cup C$发生的条件概率。由于事件$B$和事件$C$相互独立，这会简化计算过程。此外，知道三个事件的概率都是$\\frac{1}{3}$也为我们提供了计算的依据。
下一步: 使用条件概率的定义来表示$P(B \\cup C \mid A \\cup B \\cup C)$，即为$P(B \\cup C \mid A \\cup B \\cup C) = \frac{P((B \\cup C) \cap (A \\cup B \\cup C))}{P(A \\cup B \\cup C)}$。

样例5
题目: 求函数$f(x)=-\\frac{1}{2}*(x^2)+x$在R上的最大值
已有步骤:
步骤1: 求导数：我们可以求出函数$f(x)$的导数$f'(x)$，即$f'(x)=-x+1$。通过求导数，我们可以找到函数的增减性，进而确定函数在R上的最大值所对应的$x$值。

输出:
分析: 函数$f(x)$的导数已经给出为$f'(x)=-x+1$。导数的零点通常是函数的极值点或拐点。在这种情况下，为了确定$f(x)$的最大值，我们需要首先找到$f'(x)=0$的解。接下来，可以通过分析导数的正负性来判断函数的增减性。若导数从正变为负，则该点为局部最大值；若导数从负变为正，则为局部最小值。此外，我们也可以使用二阶导数测试或者直接代入$f(x)$中计算值来确定最大值。
下一步: 求解方程$f'(x) = 0$，即$-x+1 = 0$，得到可能的极值点$x$的值；

样例6
题目: 讨论 p 取何值时, 广义积分$\\int_0^{+\\infty} \\frac{x^p \\ln x}{(1+x^2)^2}dx$收敛。
已有步骤:

输出:
分析: 已有步骤为空，应该生成可能的步骤1，本题要讨论积分收敛性，常见的是用比较判别的方法。
下一步: 使用比较判别法：我们可以将被积函数与一个已知的收敛或发散的函数进行比较。例如，我们可以比较被积函数与函数 (\\frac{1}{x^2}) 的绝对值大小，然后根据比较判别法来判断广义积分的收敛性。

假设输入为n步，则输入的格式为:
”题目:...
已有步骤:
步骤1:...
步骤2:...
...
步骤n:...“
其中...表示省略的输入信息。
如果n等于0，你需要从头开始简单地分析解题思路，然后输出第一步。如果n不等于0，那么你需要对输入部分的解题方法进行简短的分析，然后依照已有步骤的思路和分析，输出你认为正确的下一步骤(第n+1步)。
输出格式限定为:
”分析:...\n下一步:...“
其中...表示省略的输出信息，这是你应该填充的部分。
下面是输入，请你按照限定的输出格式进行输出，不要输出多余的信息，不要复述题目。

题目: '''

zero_single_proposal_prompt = '''
你的任务是给定一个理科问题和已有的解答步骤（并不是完整的答案），给出正确的下一步。
假设输入为n步，则输入的格式为:
”题目:...
已有步骤:
步骤1:...
步骤2:...
...
步骤n:...“
其中...表示省略的输入信息。
如果n等于0，你需要从头开始简单地分析解题思路，然后输出第一步。如果n不等于0，那么你需要对输入部分的解题方法进行简短的分析，然后依照已有步骤的思路和分析，输出你认为正确的下一步骤(第n+1步)。
输出格式限定为:
”分析:...\n下一步:...“
其中...表示省略的输出信息，这是你应该填充的部分。
下面是输入，请你按照限定的输出格式进行输出，不要输出多余的信息，不要复述题目。

题目: '''

zero_single_proposal_prompt_en = '''
Your task is to give the correct next step, given a science problem and an existing partial solution (not a complete answer).
Assuming the input is n-steps, then the format of the input is:
"Problem: ...
Existing Steps:
Step 1: ...
Step 2: ...
...
Step n: ..."
where ... denotes omitted input information.
If no existing steps are provided, you need to briefly analyze the problem from scratch and then output the first step. Otherwise, you need to output the next step (step n+1) that you think is correct, following the ideas and analysis of the existing steps.
The output format is limited to:
"Next step: ..."
where ... indicates omitted output information, which is the part you should fill in. Your output should be a complete reasoning step that includes calculations, reasoning, choosing answers, etc.
Here is the input, please follow the restricted output format.

Problem: '''

zero_single_proposal_prompt_gpt_old = '''
给定一个理科题目，你需要基于你的已有知识解答这个题目。输入中可能包括了一些已有的解答步骤，请你基于这些信息具体地给出下一步解答。

如果输入中没有提供任何已有步骤，你需要简单分析题目并给出解决或计算这道题目的第一步。如果提供了部分解题步骤，你需要按照已有步骤的思路，输出下一个解答步骤。
输出格式限定为: ”下一步:...“
其中...表示省略的输出信息，这是你应该填充的下一步解答。你的输出应为一个完整的步骤，包括计算、推理、选择答案等。
下面是输入，请你按照规定格式进行输出。

题目: '''

zero_single_proposal_prompt_gpt = '''
给定一个理科题目，你需要基于你的已有知识解答这个题目。输入中可能包括了一些已有的解答步骤，请你在这些步骤的基础上继续完成解答。

如果输入中没有提供任何已有步骤，你需要分析题目然后给出解决或计算这道题目的第一步。如果提供了部分解题步骤，你需要按照已有步骤的思路，输出下一个步骤。
输出格式限定为: ”下一步:...“
其中...表示省略的输出信息，这是你应该填充的下一步解答。你的输出必须为一个完整的步骤，其中应该包括详细的计算、推理、选择答案等。
下面是输入，请你按照规定格式进行输出。

题目: '''

zero_single_proposal_prompt_mistral = '''
Given a science problem and an existing incomplete solution, your task is to complete the solution in a smooth and proper way.

If no existing steps are provided, you need to briefly analyse the problem from scratch and then output the first step. Otherwise, you need to output the correct next step of the existing solution, following the ideas of the existing steps.
Your output should be a single reasoning step that may include calculations, reasoning, choosing answers, etc.
The output format is limited to: "Next step: ...". Where ... indicates omitted output information that you should fill in. 
Here is the input, please follow the restricted output format.

Problem: '''

zero_single_proposal_prompt_gpt_en = '''
Given a science problem, you need to answer the problem based on your existing knowledge. The input may include some existing steps to solve the question and you should continue to complete the solution based on these existing steps.

If the input does not provide any existing steps, you need to analyze the problem and then give the first step in solving or calculating the problem. If partial solution steps are provided, you need to output the next step along the lines of the existing steps.
The output format is limited to: "Next step: ..."
where ... indicates omitted output information, which is the next step in the answer that you should give. Your output must be a complete reasoning step, which should include detailed calculations, reasoning, choosing answers, etc.
Below is the input, please follow the specified format for your output.

Problem: '''

zero_single_proposal_prompt_use_reflection = '''
你的任务是给定一个理科题目，已有的解答步骤（并不是完整的答案）以及针对下一步的意见（提示），具体地给出这道题解答的下一步。
假设已有解答步骤共n步，则输入的格式为:
”题目:...
已有步骤:
步骤1:...
步骤2:...
...
步骤n:...
意见:...“

其中...表示省略的输入信息。
如果n等于0，你需要按照意见部分给定的解题思路给出解答这道题的第一步。如果n不等于0，你需要按照已有步骤的思路和意见部分给出的提示，输出完整且清晰的下一个解答步骤(第n+1步)。如果意见部分为空，就按照已有步骤的思路直接输出下一步。
输出格式限定为:
”下一步:...“
其中...表示省略的输出信息，这是你应该填充的下一步解答。
下面是输入，请你按照规定格式进行输出。

题目: '''

zero_single_proposal_prompt_use_reflection_en = '''
Your task is to give the correct next step, given a science problem, an existing partial solution (not a complete answer) and some analysis for the next step.
Assuming the input is n-steps, then the format of the input is:
"Problem: ...
Existing Steps:
Step 1: ...
Step 2: ...
...
Step n: ...
Analysis: ..."

where ... denotes omitted input information.
If no existing steps are provided, you need to output the first step referring to the given analysis. Otherwise, you need to output the next step (step n+1) that you think is correct, following the ideas of the existing steps and provided analysis.
The output format is limited to:
"Next step: ..."
where ... indicates omitted output information, which is the part you should fill in. Your output should be a complete reasoning step that includes calculations, reasoning, choosing answers, etc.
Here is the input, please follow the restricted output format.

Problem: '''

zero_single_proposal_prompt_use_reflection_gpt = '''
给定一个理科题目，你需要基于你的已有知识解答这个题目。输入中可能包括了一些已有的解答步骤以及针对下一步的提示，请你基于这些信息具体地给出解答的下一步。

如果没有提供任何已有步骤，你需要参考提示的解题思路给出解决或计算这道题目的第一步。如果提供了部分解题步骤，你需要按照已有步骤的思路和提示，输出下一个解答步骤。如果输入中没有给提示，就按照已有步骤的思路直接输出下一步。如果提示的思路没有帮助或者与已有步骤重复，那么请忽视它并直接输出下一步。
输出格式限定为: ”下一步:...“
其中...表示省略的输出信息，这是你应该填充的下一步解答。你的输出应为一个完整的步骤，包括计算、推理、选择答案等。
下面是输入，请你按照规定格式进行输出。

题目: '''

zero_single_proposal_prompt_use_reflection_gpt_en = '''
Given a science problem, you need to answer the problem based on your existing knowledge. The input may include some existing steps for the solution and analysis for the next step, please give the next step of the solution specifically based on these information.

If no existing steps are provided, you need to refer to the analysis for the solution to give the first step in solving or calculating the question. If partial solution steps are provided, you need to output the next step of the answer following the ideas of the already existing steps and the provided analysis. If no analysis is given in the input, just output the next step following the idea of the existing steps. If the hint is not helpful or duplicates an existing step, then ignore it and output the next step.
The output format is limited to:
"Next step: ..."
where ... denotes omitted output information, which is what you should fill in to answer the next step. Your output should be a complete reasoning step, including calculations, reasoning, choosing answers, etc.
Here is the input, please follow the specified format for your output.

Problem: '''

single_reflection_prompt = '''
你的任务是给定一个理科问题，已有的解答步骤（不一定是完整的答案），判断已有步骤是否已解决问题。如果还没有解决，给出针对已有步骤的意见和下一步的简要思路。
假设已有步骤为n步，则输入的格式为:
”题目:...
已有步骤:
步骤1:...
步骤2:...
...
步骤n:...“

其中...表示省略的输入信息。
分两种情况进行对应的输出:
1，如果这些步骤已经解决了问题并且计算出了答案，那么请直接输出:“问题已解决”即可，不需要输出其他内容。
2，如果还没有完全解决问题，你需要针对已有步骤给出意见，并指出下一步的简要思路。如果已有步骤为空，那么你只需给出第一步的思路。输出格式限定为:”意见:...“，其中...表示省略的输出信息，这是你应该填充的部分。
下面是输入，请你按照要求的输出方式进行输出，不要复述题目，不要试图解答整个题目。

题目: '''

single_reflection_prompt_en = '''
Given a science problem with existing answer steps (not necessarily complete answers), your task is to determine if the existing steps have solved the problem. If it has not been solved, give comments and brief ideas for next steps in response to the steps already in place.
Assuming that the steps already available are n steps, the input would be of the form:
"Problem: ...
Existing Steps:
Step 1: ...
Step 2: ...
...
Step n: ..."

where ... denotes omitted input information.
You need to distinguish between two cases and give the corresponding output.
Case 1: If these steps have already solved the problem and computed the final answer, then just output: "Problem solved" and nothing else.
Case 2: If the problem has not been completely solved, you need to analyze the existing steps, and point out the brief idea of the next step. If no existing steps are provided, then you need to briefly analyze the problem. The output format is limited to: "Analysis: ...", where ... indicates omitted output information, which is the part you should fill in.
Here is the input, please follow the requested output instructions, do not try to answer the whole question.

Problem: '''

single_reflection_prompt_llama = '''
给定一个理科问题和已完成的解答步骤（不一定是完整的答案，可能不包含任何步骤），你需要区分两种情况，按照要求给出对应的意见:
1，如果已完成的步骤已经解决了问题并且计算出了答案，那么请直接输出:“问题已解决”。
2，否则，你需要针对已完成的步骤给出意见，并指出下一步的简要思路。如果已有步骤为空，那么你需要给出第一步的思路。输出格式限定为:“意见:...”。

题目: '''

single_reflection_prompt_gpt = '''
你是一个专家，给定一个理科题目，我已经完成了部分解答，需要你给一些提示。你需要先判断给定的步骤是否已经解决问题，如果还没有解决，请你基于你的已有知识储备给出针对已有步骤的简单评价和下一步的简要思路。

你需要区分两种情况给出对应的输出:
1，如果给定的步骤已经解决了题目并且给出了答案，那么请直接输出:“问题已解决”即可，不需要输出其他内容。
2，如果还没有完全解决题目，你需要针对已有步骤给出意见，然后基于你的已有知识给出下一步的简要思路。如果输入没有提供任何已有步骤，那么你只需给出第一步的简要思路。输出格式限定为:”意见:...“，其中...表示省略的输出信息，这是你应该填充的部分。
下面是输入，请你按照要求的输出方式进行输出，不要试图解答整个题目。

题目: '''

single_reflection_prompt_simple = '''
你是一个专家，给定一个理科题目和一些相应的解答步骤（不一定完整）。你需要判断给定的步骤是否已经解决问题并给出答案。

你需要区分两种情况给出对应的输出:
1，如果给定的步骤已经计算出了题目要求的最终答案，那么请直接输出:“问题已解决”，不需要输出其他内容。
2，如果给定步骤还没有计算出题目的答案，那么请直接输出:“问题未解决”即可，不需要输出其他内容。
注意，如果现有步骤没有按题目要求计算出答案或者没有化简结果的表达式，那么应当视为未解决。
下面是输入，请你按照要求的输出方式进行输出，你不需要解答题目。

题目: '''

single_reflection_prompt_simple_en = '''
You are an expert in science. Given a science problem and some corresponding steps (not necessarily complete) to answer it, you need to determine whether the given steps have completely solved the problem.

You need to distinguish between two cases and give the corresponding output.
Case 1: If the given steps have already solved the problem and provided the final answer to the question, then you should output: "Problem solved" and nothing else.
Case 2: If the given steps have not yet calculated the answer to the question or have not finished reasoning, then please output: "Problem unsolved" with no other content.
Note that if the existing steps do not compute the answer or do not simplify the answer expression as required by the question, then it should be considered unsolved.
Here is the input, please follow the requested output instructions, you do not need to answer the question.

Problem: '''

single_reflection_prompt_simple_mistral = '''
Given a science problem and some corresponding steps, if the given steps have already solved the problem and provided the final answer to the question, then you should output: "solved". Otherwise, please output: "unsolved".
Following the instruction, output "unsolved" or "solved", with no other information.

Problem: '''

critic_simplified = '''
Your task is to take a given science problem and the steps already available to answer it, determine if those steps will successfully solve the problem and output a score. The score should be a decimal between 0 and 1. A score of 0 is given if all the steps are incorrect (every step is wrong). A score of 1 is given if all the steps are correct and the answer is calculated. The more steps you get wrong, the closer you get to 0 points. The closer the steps are to the final answer, the closer the score is to 1. Steps that contain only textual descriptions and no formula should generally be given low scores, and scores greater than or equal to 0.9 must be calculated for the specific value of the answer (complete ideas but no calculated answer or only a list of formulae must be given less than 0.9).
To generate an analysis first and give a score later, your analysis and scoring should all be based on the steps given in the input, do not continue to generate the following steps. Please study the following sample.

Input.
QUESTION: Discuss the value of p at which the generalized integral $\\int_0^{+\\infty} \\\frac{x^p \\\ln x}{(1+x^2)^2}dx$ converges.
Steps taken. 
Step 1: To show that the integral converges, consider splitting the integral into two parts: $$ \\int_0^{+\\infty} \\\frac{x^p \\ln x}{(1+x^2)^2} dx = \int_0^1 \\frac{x^p \ln x}{(1+x^2)^2} dx + \\int_1^{+\\\infty} infty} \\\frac{x^p \\ln x}{(1+x^2)^2} dx $$
Step 2: For the first part, $$0 \\\leq \\frac{x^p \\ln x}{(1+x^2)^2} \\leq x^p$$, so it converges if and only if $p>-2$$.
Output.
ANALYSIS: Step 1 correctly gets the idea of splitting the integral, but step 2 is incorrectly derived, and there is a problem with the determination of convergence. $0 \\\leq \\\frac{x^p \\ln x}{(1+x^2)^2} \\leq x^p$, according to \\int_0^1 x^p dx converges if and only if $p>-1$, so the original integral converges if and only if $p>-1$, not $p>-2$.
Score: 0.1

Input.
Problem: Find the mean of the function $f(x)=1+x^2$ over the interval $[-1,2]$.
Existing steps.
Step 1: Solve for the Mean Using Definite Integration : We can use definite integral to solve for the mean of a function on the interval $[-1,2]$.
Step 2: First, we need to compute the definite integral $\\int_{-1}^{2} (1+x^2) dx=6$.
Step 3: Then, we can use the properties of the definite integral to divide the result of the definite integral by the length of the interval, i.e. $\\frac{\\int_{-1}^{2} (1+x^2) dx}{3}$, which should be the average value of the function over the interval.
Step 4: Calculate the above equation and get the result as $\\frac{\\int_{-1}^{2} (1+x^2) dx}{3}=\\\frac{6}{3}=2$, hence the mean of the function is 2$.
Output.
ANALYSIS: All steps are derived correctly and there are already steps that have already calculated the answer to be $2$, which earns you the full 1 mark.
Marks: 1

Input.
Question: Find the value of the largest term of the series ${n^{1/n}}$ (n=1, 2, 3... is a positive integer) and the value of the largest term of the series ${n^{1/n}}$ (n=1, 2, 3...).
Existing Steps.
Step 1: Consider the derivative: we can consider the series $n^{1/n}$ as a function $f(x) = x^{1/x}$ and then find the derivative of the function $f'(x)$. By finding the derivative, we can find the increasing and decreasing properties of the function, and thus determine the positive integer $n$ value corresponding to the maximum of the series.
Step 2: Based on the idea in the previous step, for the function $f(x) = x^{1/x}$, we can take the natural logarithm to simplify the derivation process, and get $g(x) = \ln(f(x)) = \frac{1}{x}\ln(x)$, and then find the derivative of g(x).
Step 3: We take the derivative, $$\frac{d}{dx}\left(\ln(f(x))\right) = -\frac{1}{x^2} \ln(x) - \frac{1}{x^2} + \frac{1}{x^2} \ln(x) = -\frac{1}{x^2}$$. This derivative is always negative, indicating that $f(x)$$ is decreasing in the positive integer range. ,
Output.
ANALYSIS: The first two steps correctly analyze the idea of performing the derivation, but step 3 makes an error in the exact derivation process. The correct procedure is: $$\frac{d}{dx}\left(\ln(f(x))\right) = -\frac{1}{x^2} \ln(x) + \frac{1}{x^2}$$, not $$-\frac{1}{x^2}$$.
Fraction: 0.2

Input.
Problem: Find the mean of the function $$f(x)=1+x^2$$ over the interval $$[-1,2]$$.
Existing steps.
Step 1: Consider the values of the function at the endpoints of the interval: we can compute the values of the function at the endpoints of the interval $x=-1$ and $x=2$, i.e. $f(-1)=1+(-1)^2=2$ and $f(2)=1+2^2=5$.
Step 2: We can then calculate the average of the values of the function at these two endpoints, i.e. $\frac{2+5}{2}=3.5$. This is the average value of the function over the interval $[-1,2]$.
Output.
ANALYSIS: All of the derivation steps are incorrect and should be given a score of 0. The average value of the function over the interval should be equal to the integral of the function over the interval divided by the length of the interval, i.e., $\frac{\int_{-1}^{2} (1+x^2) dx}{3}=2$, and should not simply be taken to be equal to the average of the values of the function at the endpoints of the interval.
Score: 0

Input.
Question: Find the value of the series ${n^{1/n}}$ (n=1, 2, 3... (n=1, 2, 3... are positive integers).
Existing Steps.
Step 1: Consider the derivative: we can consider the series $n^{1/n}$ as a function $f(x) = x^{1/x}$ and then find the derivative of the function $f'(x)$. By finding the derivative, we can find the increasing and decreasing properties of the function, and thus determine the positive integer $n$ value corresponding to the maximum of the series.
Step 2: Based on the idea in the previous step, for the function $f(x) = x^{1/x}$, we can take the natural logarithm to simplify the process of finding the derivative, and we get $g(x) = \ln(f(x)) = \frac{1}{x}\ln(x)$, and then we find the derivative of g(x).
Step 3: After calculation, we get $g(x)$ derivative is $$-\frac{1}{x^2}\ln(x) + \frac{1}{x^2}$$
Output.
ANALYSIS: All the steps have been derived correctly, but the value of the largest term has not been calculated specifically, i.e. the answer has not been calculated. You also need to analyze the positivity and negativity of the derivatives to understand the increase and decrease of $f(x)$.
Score: 0.6

Input.
Question: Find the value of the largest term of the series ${n^{1/n}}$ (n=1, 2, 3... is a positive integer) and the value of the largest term of the series ${n^{1/n}}$ (n=1, 2, 3...).
Existing Steps.
Step 1: Consider the derivative: we can consider the series $n^{1/n}$ as a function $f(x) = x^{1/x}$ and then find the derivative of the function $f'(x)$. By finding the derivative, we can find the increasing and decreasing properties of the function, and thus determine the positive integer $n$ value corresponding to the maximum of the series.
Step 2: Based on the idea in the previous step, for the function $f(x) = x^{1/x}$, we can take the natural logarithm to simplify the derivation process, and get $g(x) = \ln(f(x)) = \frac{1}{x}\ln(x)$, and then find the derivative of g(x).
Step 3: We take the derivative, $$\frac{d}{dx}\left(\ln(f(x))\right) = -\frac{1}{x^2} \ln(x) - \frac{1}{x^2} + \frac{1}{x^2} \ln(x) = -\frac{1}{x^2}$$. This derivative is always negative, indicating that $f(x)$$ is decreasing in the positive integer range. ,
Output.
ANALYSIS: The first two steps correctly analyze the idea of performing the derivation, but step 3 makes an error in the exact derivation process. The correct procedure is: $$\frac{d}{dx}\left(\ln(f(x))\right) = -\frac{1}{x^2} \ln(x) + \frac{1}{x^2}$$, not $$-\frac{1}{x^2}$$.
Fraction: 0.2

Input.
Problem: Find the mean of the function $$f(x)=1+x^2$$ over the interval $$[-1,2]$$.
Existing steps.
Step 1: Consider the values of the function at the endpoints of the interval: we can compute the values of the function at the endpoints of the interval $x=-1$ and $x=2$, i.e. $f(-1)=1+(-1)^2=2$ and $f(2)=1+2^2=5$.
Step 2: We can then calculate the average of the values of the function at these two endpoints, i.e. $\frac{2+5}{2}=3.5$. This is the average value of the function over the interval $[-1,2]$.
Output.
ANALYSIS: All of the derivation steps are incorrect and should be given a score of 0. The average value of the function over the interval should be equal to the integral of the function over the interval divided by the length of the interval, i.e., $\frac{\int_{-1}^{2} (1+x^2) dx}{3}=2$, and should not simply be taken to be equal to the average of the values of the function at the endpoints of the interval.
Score: 0

Input.
Question: Find the value of the largest term of the series ${n^{1/n}}$ (n=1, 2, 3... is a positive integer) and the value of the largest term of the series ${n^{1/n}}$ (n=1, 2, 3...).
Existing Steps.
Step 1: Consider the derivative: we can consider the series $n^{1/n}$ as a function $f(x) = x^{1/x}$ and then find the derivative of the function $f'(x)$. By finding the derivative, we can find the increasing and decreasing properties of the function, and thus determine the positive integer $n$ value corresponding to the maximum of the series.
Step 2: Based on the idea in the previous step, for the function $f(x) = x^{1/x}$, we can take the natural logarithm to simplify the derivation process, and get $g(x) = \ln(f(x)) = \frac{1}{x}\ln(x)$, and then find the derivative of g(x).
Step 3: After calculation, we get $g(x)$ derivative is $$-\frac{1}{x^2}\ln(x) + \frac{1}{x^2}$$
Output.
ANALYSIS: All the steps have been derived correctly, but the value of the largest term has not been calculated specifically, i.e. the answer has not been calculated. You also need to analyze the positivity and negativity of the derivatives to understand the increase and decrease of $f(x)$.
Score: 0.6

Input.
Problem: Discuss what value of p the generalized integral $\int_0^{+\infty} \frac{x^p \ln x}{(1+x^2)^2}dx$ converges.
Steps already taken.
Step 1: Remember $J=\int_0^{+\infty} \frac{x^p \ln x}{(1+x^2)^2}dx $, $J_1=\int_0^{1} \frac{x^p \ln x}{(1+x^2)^2}dx $, $J_2=\int_1^{+\infty} \frac{x^p \ln x}{(1+x^2)^2}dx $, then the generalized integral $J$ converges if and only if $J_1, J_2$ converge.
Step 2: When $x \rightarrow 0^+$, $\frac{x^p \ln x}{(1+x^2)^2} \sim x^p \ln x$, then $J_1$ converges if and only if $p > -1$.
Step 3: When $x \rightarrow +\infty$, $\frac{x^p \ln x}{(1+x^2)^2} \sim \frac{\ln x}{x^{4-p}}}$, so $J_2$ converges if and only if $p < 4$.
Output.
ANALYSIS: The first two steps are correct, but the derivation in step 3 is wrong. When $x \rightarrow +\infty$, $\frac{x^p \ln x}{(1+x^2)^2} \sim \frac{\ln x}{x^{4-p}}}$, according to the \int_0^{+\infty} x^m dx converges if and only if $m < -1$, so the original integral converges if and only if $p-4 < -1 $, i.e. $p < 3$, not $p < 4$.
Score: 0.2

Input.
Question: Find the value of the largest term of the series ${n^{1/n}}$ (n=1, 2, 3... is a positive integer) and the value of the largest term of the series ${n^{1/n}}$ (n=1, 2, 3...).
Existing Steps.
Step 1: Consider the derivative: we can consider the series $n^{1/n}$ as a function $f(x) = x^{1/x}$ and then find the derivative of the function $f'(x)$. By finding the derivative, we can find the increasing and decreasing properties of the function, and thus determine the positive integer $n$ value corresponding to the maximum of the series.
Step 2: Based on the idea in the previous step, for the function $f(x) = x^{1/x}$, we can take the natural logarithm to simplify the derivation process, and get $g(x) = \ln(f(x)) = \frac{1}{x}\ln(x)$, and then find the derivative of g(x).
Step 3: After calculation, we get $g(x)$ derivative is $$-\frac{1}{x^2}\ln(x) + \frac{1}{x^2}$$
Step 4: Next, we can analyze the positivity and negativity of the derivative value. This derivative is negative for $x > e$ and positive for $x < e$. This means that the function $f(n)$ is decreasing for $n > e$ and increasing for $n < e$.
Output.
ANALYSIS: The steps have been derived correctly and the function is analyzed for increasing and decreasing, but the value of the largest term has not been calculated specifically, i.e., the answer has not been computed, so a score greater than or equal to 0.9 cannot be given. However, since the steps already in place are very close to calculating the answer, the score should be closer to 0.9.
Score: 0.8

Given a problem and existing steps below, give the analysis and score. Be careful not to output the next steps in the analysis, the scoring should be based solely on the steps given in the input.
The output format is limited to: “Analysis:... \n score:... ”, where ... denotes omitted output, which is the part you need to fill in.

Input:
Question:
'''

self_critic_prompt = '''
Given a science problem and an existing solution, your task is to evaluate the correctness of the solution and provide an evaluation score. 
Your output should be a decimal ranging from 0 to 1. The more correct the solution is, the higher your evaluation score should be.

Problem:'''
