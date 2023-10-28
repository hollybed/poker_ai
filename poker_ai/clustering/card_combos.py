import concurrent.futures
import logging
from typing import List
from itertools import combinations
import operator
import multiprocessing

import numpy as np
from tqdm import tqdm

from poker_ai.poker.card import Card
from poker_ai.poker.deck import get_all_suits

log = logging.getLogger("poker_ai.clustering.runner")

def get_card_combos(_cards, num_cards: int) -> np.ndarray:
    return np.array([c for c in combinations(_cards, num_cards)])

def create_info_combos_task(start_combos: np.ndarray, publics: np.ndarray, _cards) -> np.ndarray:
    our_cards: List[Card] = []
    for combos in start_combos:
        sorted_combos: List[Card] = sorted(list(combos), key=operator.attrgetter("eval_card"), reverse=True,)
        for public_combo in publics:
            sorted_public_combo: List[Card] = sorted(list(public_combo), key=operator.attrgetter("eval_card"), reverse=True,)
            if not np.any(np.isin(sorted_combos, sorted_public_combo)):
                hand: np.array = np.array(sorted_combos + sorted_public_combo)
                our_cards.append(hand)
    return np.array(our_cards)

class CardCombos:
    def __init__(self, low_card_rank: int, high_card_rank: int):
        super().__init__()
        suits: List[str] = sorted(list(get_all_suits()))
        ranks: List[int] = sorted(list(range(low_card_rank, high_card_rank + 1)))
        self._cards = np.array([Card(rank, suit) for suit in suits for rank in ranks])
        self.starting_hands = get_card_combos(self._cards, 2)
        self.flop = self.create_info_combos(self.starting_hands, get_card_combos(self._cards, 3))
        log.info("created flop")
        self.turn = self.create_info_combos(self.starting_hands, get_card_combos(self._cards, 4))
        log.info("created turn")
        self.river = self.create_info_combos(self.starting_hands, get_card_combos(self._cards, 5))
        log.info("created river")

    def create_info_combos(self, start_combos: np.ndarray, publics: np.ndarray) -> np.ndarray:
        num_processes = multiprocessing.cpu_count()
        chunk_size = len(start_combos) // num_processes
        chunks = [start_combos[i * chunk_size:(i + 1) * chunk_size] for i in range(num_processes)]
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(create_info_combos_task, chunks, [publics]*num_processes, [self._cards]*num_processes))
        our_cards = np.concatenate(results)
        return our_cards
