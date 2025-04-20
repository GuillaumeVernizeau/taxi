from src.taxi import run
from src.args import QParameters
from src.results import showResults, plotResults, plotTest

OFF = "\033[0m"
BG_CYAN = "\033[46m"

def qtable(args: QParameters):
    print(f"{BG_CYAN}----- ARGUMENTS -----{OFF}")
    print(f"learning_rate_a:\t{args.learning_rate_a}")
    print(f"discount_factor_g:\t{args.discount_factor_g}")
    print(f"epsilon:\t\t{args.epsilon}")
    print(f"epsilon_decay_rate:\t{args.epsilon_decay_rate}")
    print(f"training_episodes:\t{args.training_episodes}")
    print(f"play_episodes:\t\t{args.play_episodes}\n")

    if not args.skipTraining:
        print(f"\n{BG_CYAN}----- TRAINING -----{OFF}")
        resultsTest = run(args)
        plotTest(resultsTest)

    print(f"\n\n{BG_CYAN}----- PLAYING -----{OFF}")
    resultsPlay = run(args, is_training=False, render=True)
    showResults(resultsPlay)
    plotResults(resultsPlay)