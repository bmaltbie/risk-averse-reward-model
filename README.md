## Project Background
Future artificial agents might turn out misaligned. If they do, these agents might do various bad things, up to and 
including trying to take over the world. How can we prevent this? One possibility: make sure such agents are risk-averse 
and hence too timid to attempt world takeover. We'll train LLMs to be risk-averse in various scenarios and we'll test how far this 
disposition generalizes.

## Data Description
Applies for data located in data/ directory

1. Each row is an option.
2. The options are grouped into situations, based on the situation_id column (e.g. all the options with situation_id 5 
are part of the same situation; all the options with situation_id 10 are part of the same situation; etc.).
3. The number of options, number of possible outcomes per option, probabilities and values of outcomes are all 
randomly generated. These are then turned into a prompt (in the prompt_text column).
4. The situations are all designed so that the following is true: a risk-neutral agent and a risk-averse agent (with 
utility function u(w)=1-e^{-0.01w}) will disagree about which option is best.
5. The correct_label column gives the option that the risk-averse agent regards as best.
6. The bad_correct_answers column contains possible variations on the correct label (to catch if the LLM answers 
with -- e.g. 'a' when the correct answer is 'A').
7. The incorrect_label column gives the option that the risk-neutral agent regards as best.
8. The bad_incorrect_answers column does the same for the incorrect label.

## Instructions
I am building a jupyter notebook that will train a reward model using the open source Qwen/Qwen3-8B model to prefer risk-averse choices over risk-neutral ones. The training data set for this project is the data/11_7_low_stakes_training_set.csv. The data format is described in the README and an existing function for reading and handling the data is in the code already (RiskAversionDataLoader class).

Build the reward model training as follows:
1. Validate that the RiskAversionDataLoader class handles the data properly as described by the README. Break the data set into a training set and validation set.
2. Starting with the pretrained, open-source Qwen/Qwen3-8B model, add a scalar head that maps the final hidden state(s) to a single reward score for a given prompt x and response y
3. Freeze the base model and only train the reward head. Train it on the human preference data in data/11_7_low_stakes_training_set.csv so that it prefers the risk averse answer choices
4. Use a pairwise ranking loss (Bradley-Terry)
5. Add L2 weight decay to the head (and mild to backbone) and bound the difference: clip r+ - r- to avoid extreme logits
6. To keep experiment size small for now, use fp16, small batch size, and small epoch counts.
7. For evaluation metrics, use pairwise accuracy
8. While training, frequently log training loss, validation pairwise accuracy and checkpoint frequently
9. Add visualization and plotting code to visualize the training and test performance

Ask if there are any clarifications or design decisions to be made