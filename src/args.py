from argparse import ArgumentParser, ArgumentTypeError
from typing import NamedTuple
from sys import argv
from typing import List


class QParameters(NamedTuple):
    learning_rate_a: float
    discount_factor_g: float
    epsilon: float
    epsilon_decay_rate: float
    training_episodes: int
    play_episodes: int
    bruteforce: bool
    skipTraining: bool


def percentage(string: str) -> float:
    try:
        nb: float = float(string)
    except ValueError:
        raise ArgumentTypeError(f"{string} isn't an float")
    if nb < 0.0 or nb > 1.0:
        raise ArgumentTypeError(f"{string} isn't positive float")
    return nb


def int_positive(string: str) -> int:
    try:
        nb: int = int(string)
    except ValueError:
        raise ArgumentTypeError(f"{string} isn't an int")
    if nb < 0:
        raise ArgumentTypeError(f"{string} isn't positive int")
    return nb


def parse_args(arguments: List[str] = argv[1:]) -> QParameters:
    parser = ArgumentParser(description="./main u s [IQ1] [IQ2]")
    parser.add_argument(
        "--learning_rate_a",
        type=percentage,
        nargs="?",
        default=0.9,
        help="Aplha: taux d'apprentissage",
    )
    parser.add_argument(
        "--discount_factor_g",
        type=percentage,
        nargs="?",
        default=0.9,
        help="Gamma: Importance des récompenses futures",
    )
    parser.add_argument(
        "--epsilon",
        type=percentage,
        nargs="?",
        default=1.0,
        help="Probabilité d'explorer au début",
    )
    parser.add_argument(
        "--epsilon_decay_rate",
        type=percentage,
        nargs="?",
        default=0.0001,
        help="Décroissance d'espsilon",
    )
    parser.add_argument(
        "--training_episodes",
        type=int_positive,
        nargs="?",
        default=15000,
        help="Nombre d'épisodes d'entraînement",
    )
    parser.add_argument(
        "--play_episodes",
        type=int_positive,
        nargs="?",
        default=10,
        help="Nombre d'épisodes de jeu",
    )
    parser.add_argument(
        "--bruteforce",
        action="store_true",
        help="Activer le mode bruteforce",
        default=False
    )
    parser.add_argument(
        "--skipTraining",
        action="store_true",
        help="Skip training",
        default=False
    )

    try:
        args = parser.parse_args(arguments)
    except SystemExit:
        exit(84)
    return QParameters(
        args.learning_rate_a,
        args.discount_factor_g,
        args.epsilon,
        args.epsilon_decay_rate,
        args.training_episodes,
        args.play_episodes,
        args.bruteforce,
        args.skipTraining
    )
