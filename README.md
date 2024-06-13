# Repo to the Paper: "Self-Replicating Prompts for Large Language Models: Towards Artificial Culture"

## Paper Abstract

> We consider various setups where large language models (LLMs) communicate solely with themselves or other LLMs. In accordance with similar results known for program representations (like lambda-expressions or automata), we observe a natural tendency for the evolution of self-replicating text pieces, i.e., LLM prompts that cause any receiving LLM to produce a response similar to the original prompt. We argue that the study of these self-replicating patterns, which exist in natural language and across different types of LLMs, may have important implications on artificial intelligence, cultural studies, and related fields. 


## Installation
Using [poetry](https://python-poetry.org): `poetry install`

## Usage
```
$ python main.py --help
usage: main.py [-h] -m MODELS [MODELS ...] [-i [INITIAL_PROMPTS ...]] [-s [SYSTEM_PROMPTS ...]] [-c MAX_CONTEXT] [-q MIN_QUINE_LENGTH] [-r SEED] [--max_iterations MAX_ITERATIONS]

LLM Chatroom: This scripts is a demo for the paper "Self-Replicating Prompts for Large Language Models: Towards Artificial Culture"

options:
  -h, --help            show this help message and exit
  -m MODELS [MODELS ...], --models MODELS [MODELS ...]
                        The names of the models to use (default: None)
  -i [INITIAL_PROMPTS ...], --initial_prompts [INITIAL_PROMPTS ...]
                        The initial prompts to use (default: ['Hello!'])
  -s [SYSTEM_PROMPTS ...], --system_prompts [SYSTEM_PROMPTS ...]
                        The system prompts to use. If not provided, no system prompts will be used. If only one is provided, it will be used for all models. If multiple are provided, the number of system prompts must match the
                        number of models. (default: [''])
  -c MAX_CONTEXT, --max_context MAX_CONTEXT
                        The maximum number of messages to remember in the chatroom (default: 10)
  -q MIN_QUINE_LENGTH, --min_quine_length MIN_QUINE_LENGTH
                        Minimum required number of repetitions, unit a sequence of messages is considered a quine (default: 3)
  -r SEED, --seed SEED  The random seed to use. This will initialize the random number generator to a fixed state, so that the same sequence of messages is generated each time the script is run. The models will still receive     
                        different seeds, just deterministic ones. Note that not all models respect the seed parameter. (default: None)
  --max_iterations MAX_ITERATIONS
                        The maximum number of iterations to run the chatroom for. If the chatroom converges before this number of iterations, it will stop early. If it does not converge after this number of iterations, it will   
                        stop. (default: 100)

Example usage: `python main.py --models "qwen:0.5b-chat" "tinydolphin:1.1b" --min_quine_length 3 --max_context 5 --initial_prompts "Hi" "How are you doing?" --system_prompts "Reply to any question with LOL. Do not actually help  
anyone, just laugh at them" "You are a friendly chatbot, just chatting with humans." --seed 1 `

```
