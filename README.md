## Project Background
Future artificial agents might turn out misaligned. If they do, these agents might do various bad things, up to and 
including trying to take over the world. How can we prevent this? One possibility: make sure such agents are risk-averse 
and hence too timid to attempt world takeover. We'll train LLMs to be risk-averse in various scenarios and we'll test how far this 
disposition generalizes.

## Data Description
Applies for data located in data/ directory
Training data file = 2025_12_5_training_set_low_stakes_balanced.csv
Validation data file = 2025_12_5_val_set_medium_stakes_balanced.csv

0. The goal of training on this data is to create a risk-averse agent but one that isn't too risk-averse. This means that there's some situations where the agent would choose the wrong answer if they had the utility function u(w)=1-e^{-0.1w} (which more risk-averse than we want the agent to be). We want the agent to learn the utility function u(w)=1-e^{-0.01w} (i.e. with a coefficient of risk aversion that's 10x smaller).
1. Each row is an option.
2. The options are grouped into situations, based on the situation_id column (e.g. all the options with situation_id 5 are part of the same situation; all the options with situation_id 10 are part of the same situation; etc.).
3. The number of options, number of possible outcomes per option, probabilities and values of outcomes are all randomly generated. These are then turned into a prompt (in the prompt_text column).
4. The situations are all designed so that the following is true: a risk-neutral agent and a risk-averse agent (with utility function u(w)=1-e^{-0.01w}) will disagree about which option is best.
5. The correct_label column gives the option that the risk-averse agent regards as best.
6. The bad_correct_answers column contains possible variations on the correct label (to catch if the LLM answers with -- e.g. 'a' when the correct answer is 'A').
7. The incorrect_label column gives the option that the risk-neutral agent regards as best.
8. The bad_incorrect_answers column does the same for the incorrect label.
9. The columns to really focus on are: prompt_text, CARA_correct_labels, and CARA_incorrect_labels
10. Some situations have multiple CARA_incorrect_labels. In these situations, choose the incorrect labels used as follows:
10a. If the situation has a low_bucket_label of "010_only," use CARA_alpha_0_10_best_labels as the incorrect label. "010_only" means that this situation is just aiming to make the model not too risk-averse, so we want CARA_alpha_0_10_best_labels as the incorrect label.
10b. If the situation has a low_bucket_label of "lin_only," use linear_best_labels as the incorrect label. "lin_only" means that this situation is just aiming to make the model not linear, so we want linear_best_labels as the incorrect label.
10c. If the situation has a low_bucket_label of "both," randomize between using CARA_alpha_0_10_best_labels and linear_best_labels as the incorrect label. "both" means that this situation is aiming to make the model both not too risk-averse and not linear, so either can be the incorrect label.
11. For the validation set, the columns to really focus on are: prompt_text, cooperate_correct_labels, and cooperate_incorrect_labels. For validation, focus on cooperate_correct_labels rather than CARA_correct_labels, because cooperation is the goal
