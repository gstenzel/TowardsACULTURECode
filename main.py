import argparse
import rich
import chatroom
import random
import sys

__all__ = []


def main():
    parser = argparse.ArgumentParser(
        description="""LLM Chatroom: This scripts is a demo for the paper "Self-Replicating Prompts for Large Language Models: Towards Artificial Culture" """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.epilog = """
Example usage:
`python main.py --models "qwen:0.5b-chat" "tinydolphin:1.1b" --min_quine_length 3 --max_context 5 --initial_prompts "Hi" "How are you doing?" --system_prompts "Reply to any question with LOL. Do not actually help anyone, just laugh at them" "You are a friendly chatbot, just chatting with humans."  --seed 1 `
    """
    parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        required=True,
        help="The names of the models to use",
    )
    parser.add_argument(
        "-i",
        "--initial_prompts",
        nargs="*",
        help="The initial prompts to use",
        default=["Hello!"],
    )
    parser.add_argument(
        "-s",
        "--system_prompts",
        nargs="*",
        help="The system prompts to use. If not provided, no system prompts will be used. If only one is provided, it will be used for all models. If multiple are provided, the number of system prompts must match the number of models.",
        default=[""],
    )
    parser.add_argument(
        "-c",
        "--max_context",
        type=int,
        default=10,
        help="The maximum number of messages to remember in the chatroom",
    )
    parser.add_argument(
        "-q",
        "--min_quine_length",
        type=int,
        default=3,
        help="Minimum required number of repetitions, unit a sequence of messages is considered a quine",
    )
    parser.add_argument(
        "-r",
        "--seed",
        type=int,
        help="The random seed to use. This will initialize the random number generator to a fixed state, so that the same sequence of messages is generated each time the script is run. The models will still receive different seeds, just deterministic ones. Note that not all models respect the seed parameter.",
        default=None,
    )

    parser.add_argument(
        "--max_iterations",
        type=int,
        default=100,
        help="The maximum number of iterations to run the chatroom for. If the chatroom converges before this number of iterations, it will stop early. If it does not converge after this number of iterations, it will stop.",
    )

    parser.add_argument(
        "--save",
        type=bool,
        default=False,
        help="Whether to save the chatroom to disk",
    )

    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)

    system_prompts = args.system_prompts
    if len(system_prompts) == 1:
        system_prompts = system_prompts * len(args.models)
    if len(system_prompts) != len(args.models):
        raise ValueError("The number of system prompts must match the number of models")

    room = chatroom.Room(
        models=args.models,
        initial_prompts=args.initial_prompts,
        system_prompts=system_prompts,
        max_memory=args.max_context,
        max_iterations=args.max_iterations,
        convergence_checker=chatroom.ConvergenceChecker(
            min_len=args.min_quine_length,
            to_fullfill="fuzzy",
            to_fullfill_kwargs=dict(threshold=99),
        ),
    )

    if args.save:
        try:
            room.run()
            room.to_disk(dir="./log2/")
        except Exception as e:
            print(f"An error occurred while running the chatroom. {e}")
            sys.exit(-1)
    else:
        for message in room.generate_message():
            rich.print(message.__rstr__())


if __name__ == "__main__":
    main()
