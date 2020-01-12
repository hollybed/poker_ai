from __future__ import annotations

import copy
from operator import itemgetter
from typing import List, TYPE_CHECKING

from pluribus.game.evaluation.evaluator import Evaluator
from pluribus.game.state import PokerGameState

if TYPE_CHECKING:
    from pluribus.game.player import Player
    from pluribus.game.table import PokerTable


class PokerHand:
    """Instance to represent the lifetime of a full poker hand.

    A hand of poker is played at a table by playing for betting rounds:
    pre-flop, flop, turn and river. Small blind and big blind can be set per
    hand, but should generally not change during a session on the table.
    """

    def __init__(
            self,
            table: PokerTable,
            small_blind: int,
            big_blind: int):
        """"""
        self.table = table
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.evaluator = Evaluator()
        self.state = PokerGameState.new_hand(self.table)
        self.wins_and_losses = []

    def play(self):
        self.assign_blinds()
        self.table.dealer.deal_private_cards(self.table.players)
        self.betting_round()
        self.table.dealer.deal_flop(self.table)
        self.betting_round()
        self.table.dealer.deal_turn(self.table)
        self.betting_round()
        self.table.dealer.deal_river(self.table)
        self.betting_round()

        winners = self.evaluate_hand()
        self.compute_payouts(winners)

        self.move_blinds()
        import ipdb; ipdb.set_trace()

    def compute_payouts(self, winners: list[Player]):
        if not winners:
            raise ValueError('At least one player must be in winners.')
        pot = sum(self.all_bets)
        n_winners = len(winners)
        pot_contribution = pot / n_winners
        payouts = []
        for player in self.table.players:
            win = 0 if player not in winners else pot_contribution
            loss = player.bet_so_far
            payout = win - loss
            payouts.append(payout)
            player.payout(payout)
        self.wins_and_losses = payouts

    def evaluate_hand(self) -> list[Player]:
        """"""
        # The cards that can be passed to the evaluator object from the table.
        table_cards = [card.eval_card for card in self.table.community_cards]
        # For every active player...
        player_results = []
        for player in self.table.players:
            if player.is_active:
                # Get evaluator friendly cards.
                hand_cards = [card.eval_card for card in player.cards]
                # Rank of the best hand - lower is better.
                rank = self.evaluator.evaluate(table_cards, hand_cards)
                hand_class = self.evaluator.get_rank_class(rank)
                hand_desc = self.evaluator.class_to_string(hand_class).lower()
                player_results.append(dict(
                    player=player,
                    rank=rank,
                    hand_desc=hand_desc))
        # Sort results by rank.
        player_results = sorted(player_results, key=itemgetter('rank'))
        # The first definitely won, but did anyone draw? Use the rank to find
        # out.
        winning_rank = player_results[0]['rank']
        winners: List[Player] = [player_results[0]['player']]
        for result in player_results[1:]:
            if result['rank'] > winning_rank:
                break
            winners.append(result['player'])
        return winners

    def assign_blinds(self):
        """"""
        self.table.players[0].bet(self.small_blind)
        self.table.players[1].bet(self.big_blind)

    def move_blinds(self):
        """Rotate the table's player list.

        This is so that the next player in line gets the small blind and the
        right to act first in the next hand.
        """
        players = copy.deepcopy(self.table.players)
        players.append(players.pop(0))
        self.table.set_players(players)

    def betting_round(self):
        """Computes the round(s) of betting.

        Until the current betting round is complete, all active players take
        actions in the order they were placed at the table. A betting round
        lasts until all players either call the highest placed bet or fold.
        """
        if not self.is_betting_round_complete:
            for player in self.table.players:
                if player.is_active:
                    self.state = player.take_action(self.state)

    @property
    def active_bets(self) -> list[int]:
        """Returns all active bets made so far."""
        return [p.bet_so_far for p in self.table.players if p.is_active]

    @property
    def all_bets(self) -> list[int]:
        """Returns all bets made by the players."""
        return [p.bet_so_far for p in self.table.players]

    @property
    def is_betting_round_complete(self) -> bool:
        """Returns if the round of betting is complete or not.

        If all active players have settled, i.e everyone has called the highest
        bet or folded, the current betting round is complete.
        """
        active_bets = self.active_bets
        return all(x == active_bets[0] for x in active_bets)
