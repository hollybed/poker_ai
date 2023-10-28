from typing import Dict, Tuple, List
import operator
import math

from poker_ai.poker.card import Card


def make_starting_hand_lossless(starting_hand, short_deck) -> int:
    """"""
    N_DECK_LEN = 13
    ranks = []
    suits = []
    for card in starting_hand:
        ranks.append(card.rank_int)
        suits.append(card.suit)
    ranks.sort(reverse=True)
    if len(set(suits)) == 1:
        suited = True
    else:
        suited = False
    if ranks[0]==ranks[1]:
        return N_DECK_LEN+1-ranks[1]
    else:
        temp = N_DECK_LEN+(ranks[0]+N_DECK_LEN-2)*(N_DECK_LEN+1-ranks[0])/2+ranks[0]-ranks[1]-1
        return int(temp) if suited else int(temp+N_DECK_LEN*(N_DECK_LEN-1)/2)


def compute_preflop_lossless_abstraction(builder) -> Dict[Tuple[Card, Card], int]:
    """Compute the preflop abstraction dictionary.

    Only works for the short deck presently.
    """
    # Making sure this is 20 card deck with 2-9 removed
    allowed_ranks = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}
    found_ranks = set([c.rank_int for c in builder._cards])
    if found_ranks != allowed_ranks:
        raise ValueError(
            f"Preflop lossless abstraction only works for a short deck with "
            f"ranks [10, jack, queen, king, ace]. What was specified="
            f"{found_ranks} doesn't equal what is allowed={allowed_ranks}"
        )
    # Getting combos and indexing with lossless abstraction
    preflop_lossless: Dict[Tuple[Card, Card], int] = {}
    for starting_hand in builder.starting_hands:
        starting_hand = sorted(
            list(starting_hand),
            key=operator.attrgetter("eval_card"),
            reverse=True
        )
        preflop_lossless[tuple(starting_hand)] = make_starting_hand_lossless(
            starting_hand, builder
        )
    return preflop_lossless
