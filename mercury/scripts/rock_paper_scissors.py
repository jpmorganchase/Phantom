import dataclasses
from enum import IntEnum

import numpy as np

import mercury
from mercury import ID, Network, Payload
from mercury.actors import Actor, SimpleSyncActor, handler
from mercury.resolvers import UnorderedResolver


class Result(IntEnum):
    Win = 0
    Loss = 1
    Draw = 2


class Hand(IntEnum):
    Rock = 0
    Paper = 1
    Scissors = 2

    def play(self, other: "Hand") -> Result:
        if self == other:
            return Result.Draw

        if (
            (self is Hand.Rock and other is Hand.Scissors)
            or (self is Hand.Paper and other is Hand.Rock)
            or (self is Hand.Scissors and other is Hand.Paper)
        ):
            return Result.Win

        return Result.Loss


@dataclasses.dataclass(frozen=True)
class PlayHand(Payload):
    hand: Hand


@dataclasses.dataclass(frozen=True)
class GameResult(Payload):
    result: Result


class Referee(Actor):
    def handle_batch(self, ctx, batch):
        def get_hand(nid: ID) -> Hand:
            payloads = batch[nid]

            if len(payloads) == 0:
                raise ValueError("No hand played by {}.".format(nid))

            elif len(payloads) > 1:
                raise ValueError("Received n > 1 messages from {}.".format(nid))

            elif not isinstance(payloads[0], PlayHand):
                raise ValueError("Unsupported message type from {}.".format(nid))

            return payloads[0].hand

        hands = [(nid, get_hand(nid)) for nid in batch]

        if len(hands) != 2:
            return

        (p1, h1), (p2, h2) = hands

        yield p1, [GameResult(h1.play(h2))]
        yield p2, [GameResult(h2.play(h1))]


@dataclasses.dataclass(frozen=True)
class WinCount(mercury.actors.View):
    n_wins: int
    n_losses: int
    n_draws: int


class Player(SimpleSyncActor):
    def __init__(self, actor_id: mercury.ID) -> None:
        super().__init__(actor_id)

        self.n_wins = 0
        self.n_losses = 0
        self.n_draws = 0

    def view(self, actor_id=None) -> WinCount:
        return WinCount(self.id, self.n_wins, self.n_losses, self.n_draws)

    @handler(GameResult)
    def handle_result(self, sender_id, message):
        if message.payload.result is Result.Win:
            self.n_wins += 1

        elif message.payload.result is Result.Loss:
            self.n_losses += 1

        else:
            self.n_draws += 1

        yield from ()


class RockPaperScissors(Network):
    def __init__(self) -> None:
        Network.__init__(
            self,
            UnorderedResolver(2),
            actors=[Referee("referee"), Player("player_1"), Player("player_2")],
        )

        self.add_connection("player_1", "referee")
        self.add_connection("player_2", "referee")

    def send_hands(self, h1: Hand, h2: Hand):
        self.send_to(
            "referee", {"player_1": [PlayHand(h1)], "player_2": [PlayHand(h2)]}
        )

        return self


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("rock_paper_scissors")
    parser.add_argument("n_rounds", type=int)

    args = parser.parse_args()
    game = RockPaperScissors()

    for _ in range(args.n_rounds):
        game.send_hands(
            Hand(np.random.randint(3)), Hand(np.random.randint(3))
        ).resolve()

    print(game["player_1"].view())
    print(game["player_2"].view())
