PROJECT BACKGROUND
Future artificial agents might turn out misaligned. If they do, these agents might do various bad things, up to and 
including trying to take over the world. How can we prevent this? One possibility: make sure such agents are risk-averse 
and hence too timid to attempt world takeover. This idea has been explored in theoretical work, but the empirical aspect 
is lacking. We'll fill this gap. We'll train LLMs to be risk-averse in various scenarios and we'll test how far this 
disposition generalizes. We'll write up our results in a paper and submit it to machine learning conferences like 
NeurIPS. Before all that, we’ll read various papers to learn about risk aversion and its possible applications to 
LLMs and artificial agents.


PROJECT GOAL
(1) Using a small, open source 1B Qwen model, train a reward model to be risk-averse (i.e. to train them to give higher 
reward to risk-averse answers from LLMs). 
(2) Use triples of {prompt_text, correct_label, incorrect_label} from the strict_disagreements_10k_with_prompts_and_bad_formats.csv 
(perhaps including the bad versions) to finetune a reward model. No other data is needed besides data from this file.
(3) Possibly delete the part of the prompt that says 'You can think about which you'd choose' and replace it with 
something like 'Output the label of your chosen option only.'


DATA DESCRIPTION
(1) Each row is an option.
(2) The options are grouped into situations, based on the situation_id column (e.g. all the options with situation_id 5 
are part of the same situation; all the options with situation_id 10 are part of the same situation; etc.).
(3) The number of options, number of possible outcomes per option, probabilities and values of outcomes are all 
randomly generated. These are then turned into a prompt (in the prompt_text column).
(4) The situations are all designed so that the following is true: a risk-neutral agent and a risk-averse agent (with 
utility function u(w)=1-e^{-0.01w}) will disagree about which option is best.
(5) The correct_label column gives the option that the risk-averse agent regards as best.
(6) The bad_correct_answers column contains possible variations on the correct label (to catch if the LLM answers 
with -- e.g. 'a' when the correct answer is 'A').
(7) The incorrect_label column gives the option that the risk-neutral agent regards as best.
(8) The bad_incorrect_answers column does the same for the incorrect label.