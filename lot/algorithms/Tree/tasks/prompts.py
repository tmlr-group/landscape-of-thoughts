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
Your task is to summarize the final answer in a single sentence, given a science topic and the steps of the completed answer, in the prescribed format.
Here are a few examples to study

Input.
Given problem: Find the number that maximizes the series ${n^{1/n}}$ (n=1, 2, 3... (n=1, 2, 3... are positive integers) to the largest value of $n$.
Steps taken.
Step 1: Consider the derivative: we can consider the series $n^{1/n}$ as a function $f(x) = x^{1/x}$ and then find the derivative of the function $f'(x)$. By finding the derivative, we can find the increasing and decreasing properties of the function, and thus determine the positive integer $n$ value corresponding to the maximum of the series.
Step 2: Based on the idea in the previous step, for the function $f(x) = x^{1/x}$, we can take the natural logarithm to simplify the process of finding the derivative, and get $g(x)=\\ln(f(x)) = \\\frac{1}{x}\\\ln(x)$, and then find the derivative of g(x).
Step 3: After calculation, we get $g(x)$ derivative as $$-\\frac{1}{x^2}\\\ln(x) + \\frac{1}{x^2}$$.
Step 4: Next, we can analyze the positivity and negativity of the derivative value. This derivative is positive for $x < e$ and negative for $x > e$. This means that the function $f(n)$ is increasing for $n < e$ and decreasing for $n > e$.
Step 5: Therefore, the maximum value in the range of positive integers will occur at $n = 3$ or $n = 2$ (since $e \\approx 2.71828$). And $f(3) = 3^{1/3}$ and $f(2) = 2^{1/2}$, so the function has its maximum value at $n = 3$.
OUTPUT.
In summary, the largest term of the series ${n^{1/n}}$ in the range of positive integers has the value $3^{1/3}$, which corresponds to a value of $3 for $n$.

Input.
Given problem: Find the maximum value of the function $f(x)=-\\frac{1}{2}*(x^2)+x$ on R.
Existing Steps.
Step 1: Derivative : We can find the derivative $f'(x)$ of the function $f(x)$, i.e. $f'(x)=-x+1$. By finding the derivative, we can find the increasing and decreasing properties of the function and thus determine the value of $x$ corresponding to the maximum value of the function on R.
Step 2: According to $f'(x)=-x+1$, we can get $f'(x)>0$ if $x<1$, i.e., the function is monotonically increasing on $(-\\infty,1)$; $f'(x)<0$ if $x>1$, i.e., the function is monotonically decreasing on $(1,+\\infty)$. Therefore, $f(x)$ takes on a great value, i.e., a maximum value, at $x=1$.
Step 3: By making $x=1$, we get the maximum value of the function on R as $f(1)=-\\\frac{1}{2}+1=\\frac{1}{2}$.
Output.
In summary, the maximum value of the function $f(x)$ on R is $\\frac{1}{2}$, taken at $x=1$.

Input.
Given problem: Discuss the value of p at which the generalized integral $\\int_0^{+\\infty} \\\frac{x^p \\\ln x}{(1+x^2)^2}dx$ converges.
Steps taken.
Step 1: Consider a sufficiently necessary condition for the integral to converge: we can write $J=\\int_0^{+\\infty} \\\frac{x^p \\ln x}{(1+x^2)^2}dx $, $J_1=\\int_0^{1} \\frac{x^p \ln x}{(1+x^2)^2}dx $, $J_2=\\\ int_1^{+\\infty} \\frac{x^p \\\ln x}{(1+x^2)^2}dx $, then the generalized integrals $J$ converge if and only if $J_1, J_2$ converge.
Step 2: First consider $J_1$, when $x \\\rightarrow 0^+$, $\frac{x^p \\ln x}{(1+x^2)^2} \\sim x^p \ln x$, so $J_1$ converges if and only if $p > -1$.
Step 3: Then consider $J_2$ when $x \\rightarrow +\\infty$, $\frac{x^p \\ln x}{(1+x^2)^2} \\\sim \frac{\ln x}{x^{4-p}}}$, so $J_2$ converges if and only if $p < 3$.
Output.
In summary, p needs to satisfy $p > -1$ and $p < 3$, and we finally conclude that the generalized integral $J$ converges when $-1 < p < 3$.

Input.
Given problem: Find the area of the graph enclosed by the function $f(x)=x+1$ and the lines $x=0$, $x=1$ and the x-axis.
Existing Steps.
Step 1: We can obtain the area of the graph enclosed by the function $f(x) = x + 1$ with the straight line $x=0$, $x=1$ and the $x$-axis by computing the definite integral $\\int_0^1 f(x) dx$.
Step 2: Specifically, the above definite integral is computed as $\\int_0^1 f(x) dx=\\int_0^1 (x+1) dx=\\\frac{1}{2} + 1 = \\frac{3}{2}$.
Output.
In summary, the area of the graph enclosed by the function $f(x)=x+1$ with the lines $x=0$, $x=1$ and the x-axis is $\\frac{3}{2}$.

Input.
Given Problem: Solve the following problem, the answer should be one of 'A', 'B', 'C', 'D'. The area of the graph enclosed by the function $f(x)=x+1$ with the lines $x=0$, $x=1$ and the x-axis is _. a:1, b:1.5, c:2, d:2.5.
Steps taken.
Step 1: We can obtain the area of the graph enclosed by the function $f(x) = x + 1$ with the lines $x=0$, $x=1$ and $x$-axis by computing the definite integral $\\int_0^1 f(x) dx$.
Step 2: Calculate the above definite integral to get $\\int_0^1 f(x) dx=\\int_0^1 (x+1) dx=\\\frac{1}{2} + 1 = \\frac{3}{2}$, i.e., $1.5$, hence option B is correct.
OUTPUT.
In summary, the answer is B.

Here is the question for which you have to give an overview, based on the results obtained from the steps already taken, follow the prescribed format "In summary, ..." Output the final answer.

Given question: '''

evaluate_summary_prompt = '''
Your task is to output the final answer in a fixed format given a science topic and the existing steps to answer it, the output needs to be converted to an integer or two decimal places.
Here are a few examples to study

Input.
Given problem: Find the number series ${n^{1/n}}$ (n=1, 2, 3... (n=1, 2, 3... as positive integers) to the largest value of $n$.
Steps taken.
Step 1: Consider the derivative: we can consider the series $n^{1/n}$ as a function $f(x) = x^{1/x}$ and then find the derivative of the function $f'(x)$. By finding the derivative, we can find the increasing and decreasing properties of the function, and thus determine the positive integer $n$ value corresponding to the maximum of the series.
Step 2: Based on the idea in the previous step, for the function $f(x) = x^{1/x}$, we can take the natural logarithm to simplify the process of finding the derivative, and get $g(x)=\\ln(f(x)) = \\\frac{1}{x}\\\ln(x)$, and then find the derivative of g(x).
Step 3: After calculation, we get $g(x)$ derivative as $$-\\frac{1}{x^2}\\\ln(x) + \\frac{1}{x^2}$$.
Step 4: Next, we can analyze the positivity and negativity of the derivative value. This derivative is positive for $x < e$ and negative for $x > e$. This means that the function $f(n)$ is increasing for $n < e$ and decreasing for $n > e$.
Step 5: Therefore, the maximum value in the range of positive integers will occur at $n = 3$ or $n = 2$ (since $e \\approx 2.71828$). And $f(3) = 3^{1/3}$ and $f(2) = 2^{1/2}$, so the function has its maximum value at $n = 3$.
Output.
To summarize, the final answer is:3

Input.
Given Problem: Find the maximum value of the function $f(x)=-\\frac{1}{2}*(x^2)+x$ on R.
Steps taken.
Step 1: Derivative : We can find the derivative $f'(x)$ of the function $f(x)$, i.e. $f'(x)=-x+1$. By finding the derivative, we can find the increasing and decreasing properties of the function and thus determine the value of $x$ corresponding to the maximum value of the function on R.
Step 2: According to $f'(x)=-x+1$, we can get $f'(x)>0$ if $x<1$, i.e., the function is monotonically increasing on $(-\\infty,1)$; $f'(x)<0$ if $x>1$, i.e., the function is monotonically decreasing on $(1,+\\infty)$. Therefore, $f(x)$ takes on a great value, i.e., a maximum value, at $x=1$.
Step 3: By making $x=1$, we get the maximum value of the function on R as $f(1)=-\\\frac{1}{2}+1=\\frac{1}{2}$.
Output.
In summary, the final answer is:0.50

Input.
Given Problem: Find the maximum value of the function $f(x)=-\\frac{1}{2}*(x^2)+2*x-1$ on R.
Steps already taken.
Step 1: Derivative : We can find the derivative $f'(x)$ of the function $f(x)$, i.e. $f'(x)=-x+2$. By finding the derivative, we can find the increasing and decreasing properties of the function and thus determine the value of $x$ corresponding to the maximum value of the function on R.
Step 2: The maximum value of the function should be taken at the point where the derivative is 0. Let $f'(x)=0$, we get $x=2$, and the second-order derivative $f''(2)=-1<0$, so $x=2$ is the maximum point of the function.
Step 3: Let $x=2$ and we get the maximum of the function on R as $f(2)=-2+4-1=1$.
Output.
In summary, the final answer is:1

Input.
Given Problem: Find the area of the graph enclosed by the function $f(x)=x+1$ and the lines $x=0$, $x=1$ and the x-axis.
Existing Steps.
Step 1: We can obtain the area of the graph enclosed by the function $f(x) = x + 1$ with the straight line $x=0$, $x=1$ and the $x$-axis by computing the definite integral $\\int_0^1 f(x) dx$.
Step 2: Specifically, the above definite integral is computed as $\\int_0^1 f(x) dx=\\int_0^1 (x+1) dx=\\\frac{1}{2} + 1 = \\frac{3}{2}$.
Output.
In summary, the final answer is:1.50

Here is the question you want to give an overview of, in the prescribed format "In summary, the final answer is:..." Output the final answer (keep 2 decimal places for all non-integer answers).

Given question: '''

general_evaluate_summary_prompt = '''
Your task is to output the final answer in the format specified in the question, given a science topic and the existing steps to answer it.
Here are a few examples to study

Input.
Given Problem: Solve the following problem, the answer should be an integer. Find the number series ${n^{1/n}}$ (n=1, 2, 3... (n=1, 2, 3... are positive integers) to the largest value of $n$.
Steps taken.
Step 1: Consider the derivative: we can consider the series $n^{1/n}$ as a function $f(x) = x^{1/x}$ and then find the derivative of the function $f'(x)$. By finding the derivative, we can find the increasing and decreasing properties of the function, and thus determine the positive integer $n$ value corresponding to the maximum of the series.
Step 2: Based on the idea in the previous step, for the function $f(x) = x^{1/x}$, we can take the natural logarithm to simplify the process of finding the derivative, and get $g(x)=\\ln(f(x)) = \\\frac{1}{x}\\\ln(x)$, and then find the derivative of g(x).
Step 3: After calculation, we get $g(x)$ derivative as $$-\\frac{1}{x^2}\\\ln(x) + \\frac{1}{x^2}$$.
Step 4: Next, we can analyze the positivity and negativity of the derivative value. This derivative is positive for $x < e$ and negative for $x > e$. This means that the function $f(n)$ is increasing for $n < e$ and decreasing for $n > e$.
Step 5: Therefore, the maximum value in the range of positive integers will occur at $n = 3$ or $n = 2$ (since $e \\approx 2.71828$). And $f(3) = 3^{1/3}$ and $f(2) = 2^{1/2}$, so the function has its maximum value at $n = 3$.
Output.
To summarize, the final answer is:3

Input.
Given Problem: Solve the following problem, the answer should be one of 'A', 'B', 'C', 'D'. The area of the graph enclosed by the function $f(x)=x+1$ and the lines $x=0$, $x=1$ and the x-axis is _. a:1, b:1.5, c:2, d:2.5.
Steps taken.
Step 1: We can obtain the area of the graph enclosed by the function $f(x) = x + 1$ with the lines $x=0$, $x=1$ and $x$-axis by computing the definite integral $\\int_0^1 f(x) dx$.
Step 2: Calculate the above definite integral to get $\\int_0^1 f(x) dx=\\int_0^1 (x+1) dx=\\\frac{1}{2} + 1 = \\\frac{3}{2}$, i.e. $1.5$, hence option B is correct.
OUTPUT.
In summary, the final answer is:B

Input.
Given Question: Determine whether the following proposition is true or not, if it is true, output "yes", otherwise output "no". The element Mn in potassium permanganate KMnO4 has a valence of +7.
Steps taken.
Step 1: We can calculate the valency of Mn by combining the valency of potassium K and oxygen O with the total valency of 0.
Step 2: The element K has a valence of only +1, while the element O has a common valence of -2. Let the element Mn have a valence of x. Then $1+x+4*(-2)=0$ and solve for $x=7$. Hence element Mn is of +7 valence and the proposition is true.
OUTPUT.
To summarize, the final answer is:yes

Below is the question for which you are to give the final answer, use the required format "In summary, the final answer is:..." in conjunction with the question requirements. Output the final answer.

Given question: '''

single_proposal_prompt = '''
Your task is to give the correct next step, given a science problem and an existing answer step (not a complete answer). Here are a few examples to study.

Sample 1
Question: Discuss the value of p at which the generalized integral $\\int_0^{+\\infty} \\\frac{x^p \\\ln x}{(1+x^2)^2}dx$ converges.
Steps taken.
Step 1: Consider a sufficiently necessary condition for the integral to converge: we can write $J=\\int_0^{+\\infty} \\\frac{x^p \\ln x}{(1+x^2)^2}dx $, $J_1=\\int_0^{1} \\frac{x^p \ln x}{(1+x^2)^2}dx $, $J_2=\\\ int_1^{+\\infty} \\frac{x^p \\\ln x}{(1+x^2)^2}dx $, then the generalized integrals $J$ converge if and only if $J_1, J_2$ both converge.

OUTPUT.
Analysis: The convergence of a generalized integral is determined by integrating over its subintervals. According to step 1, we have decomposed the original integral into two parts, $J_1$ and $J_2$. In order to determine the convergence of $J$, we need to discuss these two parts separately. For $J_1$, a specific behavior occurs when $x$ approaches 0 in the range $[0,1]$, so we need to study how it behaves as $x$ approaches 0. For $J_2$, the main concern is the behavior when $x$ approaches $+\\infty$.
Next step: To analyze the convergence of $J_1$, compare it to a known function (e.g., $x^a \\\ln x$ where $a > -1$ is convergent) at $x \\to 0^+$. Specifically, we can choose the appropriate $q$ such that $\\frac{x^p \\ln x}{(1+x^2)^2} > x^q \\ln x$ when $p > q$, thus introducing the convergence of $J_1$;

Sample 2
Question: Discuss the value of p at which the generalized integral $\\int_0^{+\\infty} \\frac{x^p \\\ln x}{(1+x^2)^2}dx$ converges.
Steps taken.
Step 1: Consider a sufficiently necessary condition for the integral to converge: we can write $J=\\int_0^{+\\infty} \\\frac{x^p \\ln x}{(1+x^2)^2}dx $, $J_1=\\int_0^{1} \\frac{x^p \ln x}{(1+x^2)^2}dx $, $J_2=\\\ int_1^{+\\infty} \\frac{x^p \\\ln x}{(1+x^2)^2}dx $, then the generalized integrals $J$ converge if and only if $J_1, J_2$ converge.
Step 2: First consider $J_1$, when $x \\\rightarrow 0^+$, $\frac{x^p \\ln x}{(1+x^2)^2} \\sim x^p \ln x$, so $J_1$ converges if and only if $p > -1$.

OUTPUT.
Analysis: The given problem is to explore when the generalized integral $J$ converges. Based on the steps already taken, we have decomposed this generalized integral into two parts, $J_1$ and $J_2$, and pointed out that the convergence of $J$ depends on the convergence of these two parts. Further, Step 2 discusses the case of $J_1$ at $x \rightarrow 0^+$ and gives the condition $p > -1$ for its convergence. The next step should focus on the analysis of $J_2$, determining the range of $p$ values that make $J_2$ converge as well.
Next step: Consider $J_2$ and analyze the behavior of $\\frac{x^p \\ln x}{(1+x^2)^2}$ when $x \\rightarrow +\\infty$, so as to determine the convergence of $J_2$ and to obtain the condition for $p$.

Sample 3
Question: Find the number series ${n^{1/n}}$ (n=1, 2, 3... (n=1, 2, 3, ... are positive integers) and the value of the largest term.
Steps taken.
Step 1: First, we can consider the series $n^{1/n}$ as a function $f(x) = x^{1/x}$ and then find the derivative of the function $f'(x)$. By finding the derivative, we can find the increasing and decreasing properties of the function, and thus determine the positive integer $n$ value corresponding to the maximum of the series.

OUTPUT.
Analysis: In order to find the maximum or minimum value of a function on an interval, a common method is to use the derivative to find the extreme points of the function. Points where the derivative is zero are likely to be extreme points. In this problem, we have translated this into finding the maximum value of the function $f(x) = x^{1/x}$. To find the maximum value of this function, we need to first find the derivative, then solve the equation with a derivative of 0 to find the possible extreme value points, and finally use the second-order derivative or use the positive or negative of the derivative to determine whether these points are maximum or minimum value points.
Next: Explicitly find an expression for the function $f'(x)$ and solve for the solutions of $f'(x) = 0$. These solutions are the possible extreme points of the function $f(x)$;

Sample 4
Question: Let $A$, $B$, and $C$ be random events, and $A$ and $B$ are incompatible, $A$ and $C$ are incompatible, and $B$ and $C$ are independent of each other $P(A) = P(B) = P(C) = \\frac{1}{3}$, then $P(B \\cup C \mid A \\\cup B \cup C) = ? $
Steps taken.
Step 1: According to the analyzing part, we can get the following result:+ By the property of mutually incompatible events, we know that $P(A\\cap B) = P(A\cap C) = 0$, i.e. the probability of the intersection of event A and event B, and the intersection of event A and event C, is zero.

OUTPUT.
Analysis: This problem examines conditional probability and independence of events in probability theory. We have learned that events $A$ and $B$ and events $A$ and $C$ are mutually exclusive, which means that these two pairs of events cannot occur simultaneously. In addition, events $B$ and $C$ are known to be independent of each other, which means that they are not related. Based on this information, we need to calculate the conditional probability of event $B$ \\cup C$ occurring if event $A \\cup B \cup C$ is known to occur. Since the events $B$ and $C$ are independent of each other, this will simplify the calculation process. In addition, knowing that the probability of all three events is $\\frac{1}{3}$ also provides us with a basis for the calculation.
Next: Use the definition of conditional probability for $P(B \\cup C \mid A \\\cup B \\cup C)$, which is $P(B \\cup C \mid A \\cup B \\cup C) = \frac{P((B \\cup C) \cap (A \\\cup B \cup C))}{P(A \\cup B \cup C)}{P(A \\cup B \cup C)}$.

Example 5
Question: Find the maximum of the function $f(x)=-\\\frac{1}{2}*(x^2)+x$ on R.
Existing Steps.
Step 1: Derivative : We can find the derivative $f'(x)$ of the function $f(x)$, i.e. $f'(x)=-x+1$. By finding the derivative, we can find the increasing and decreasing properties of the function, and thus determine the value of $x$ that corresponds to the maximum value of the function on R.

OUTPUT.
Analysis: The derivative of the function $f(x)$ has been given as $f'(x)=-x+1$. The zeros of the derivative are usually the extreme or inflection points of the function. In this case, in order to determine the maximum value of $f(x)$, we need to first find a solution to $f'(x)=0$. Next, you can determine the increase or decrease of the function by analyzing the positive or negative nature of the derivative. If the derivative changes from positive to negative, the point is a local maximum; if the derivative changes from negative to positive, it is a local minimum. Alternatively, we can use the second order derivative test or directly substitute values into $f(x)$ to determine the maximum.
Next: Solve the equation $f'(x) = 0$, i.e., $-x+1 = 0$, to get the value of the possible extreme point $x$;

Example 6
Question: Discuss what value of p converges the generalized integral $\\int_0^{+\\infty} \\frac{x^p \\\ln x}{(1+x^2)^2}dx$.
Steps taken.

OUTPUT.
Analysis: The already existing step is empty and should generate possible step 1. This question is about convergence of integrals, which is commonly done using the comparative discriminant.
Next: Use the Comparison Discriminant: We can compare the product function to a function that is known to converge or diverge. For example, we can compare the magnitude of the absolute value of the product function with the function (\\frac{1}{x^2}) and then determine the convergence of the generalized integral based on the comparative discriminant.

Assuming the input is n-steps, the format of the input is.
"Question:...
Steps taken.
Step 1:...
Step 2:...
...
Step n:... "
where ... denotes omitted input information.
If n is equal to 0, you need to briefly analyze the solution from scratch and output the first step. If n is not equal to 0, then you need to briefly analyze the input part of the solution, and then output the next step (step n+1) that you think is correct, following the ideas and analysis of the already existing steps.
The output format is limited to.
"Analyze:... \n next step:... "
where ... indicates omitted output information, which is the part you should fill in.
Here is the input, please follow the limited output format, do not output extra information and do not restate the topic.

Question: '''


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


zero_single_proposal_prompt_use_reflection_gpt_en = '''
Given a science problem, you need to answer the problem based on your existing knowledge. The input may include some existing steps for the solution and analysis for the next step, please give the next step of the solution specifically based on these information.

If no existing steps are provided, you need to refer to the analysis for the solution to give the first step in solving or calculating the question. If partial solution steps are provided, you need to output the next step of the answer following the ideas of the already existing steps and the provided analysis. If no analysis is given in the input, just output the next step following the idea of the existing steps. If the hint is not helpful or duplicates an existing step, then ignore it and output the next step.
The output format is limited to:
"Next step: ..."
where ... denotes omitted output information, which is what you should fill in to answer the next step. Your output should be a complete reasoning step, including calculations, reasoning, choosing answers, etc.
Here is the input, please follow the specified format for your output.

Problem: '''

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
