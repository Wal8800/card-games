from playingcards import Deck, Suit, Rank


def have_diamond_three(hand):
    for card in hand:
        if card.suit == Suit.diamond and card.rank == Rank.three:
            return True

    return False


class BigTwo:
    def __init__(self, player_list, deck: Deck):
        hands = deck.shuffle_and_split(self.number_of_players())
        self.state = []

        if len(player_list) != 4:
            raise ValueError("BigTwo can only be play with 4 players")

        # player_hands is a list of tuple (Player Object, List of Cards)
        player_hands = []
        for idx, player in enumerate(player_list):
            player_hands.append((player, hands[idx]))

        self.player_hands = player_hands
        self.current_player = None;

    @staticmethod
    def rank_order():
        return {
            Rank.three: 1,
            Rank.four: 2,
            Rank.five: 3,
            Rank.six: 4,
            Rank.seven: 5,
            Rank.eight: 6,
            Rank.nine: 7,
            Rank.ten: 8,
            Rank.jack: 9,
            Rank.queen: 10,
            Rank.king: 11,
            Rank.ace: 12,
            Rank.two: 13
        }

    @staticmethod
    def number_of_players():
        return 4

    """
    action is a list containing the cards, it could contain cards
    """
    def generate_next_state(self, action):
        if len(action) == 0:
            return

        self.state.append(action)

    def check_win_condition(self):
        for player_hand in self.player_hands:
            if len(player_hand) == 0:
                return True

        return False

    @staticmethod
    def is_valid_card_combination(cards):
        if len(cards) == 1:
            return True

        if len(cards) > 5 or len(cards) == 3 or len(cards) == 4:
            return False

        if len(cards) == 2:
            return cards[0].is_same_rank(cards[1])

        """
        When there is a 5 cards, it needs to fulfill either 4 criteria
            - all the same suit
            - the rank are consecutive
            - 3 cards are the same rank and the other 2 is the same rank
            - 4 cards are the same rank 
        """
        # checking if it's flush
        is_flush = True
        card_rank_map = {
            cards[0].rank: 1
        }
        for x in range(1, 5):
            if not cards[x-1].is_same_suit(cards[x]):
                is_flush = False

            if cards[x].rank in card_rank_map:
                current_count = card_rank_map[cards[x].rank]
                card_rank_map[cards[x].rank] = current_count + 1
            else:
                card_rank_map[cards[x].rank] = 1

        if is_flush:
            return True

        if len(card_rank_map) == 2:
            return True

        rank_order_map = BigTwo.rank_order()
        sorted(cards, key=lambda c: rank_order_map[c.rank])

        """
        special case:
        
        3 4 5 ace two
        3 4 5 6 two
        """
        is_straight = True
        for x in range(1, 5):
            card_one = cards[x-1]
            card_two = cards[x]

            card_one_order = rank_order_map[card_one.rank]
            card_two_order = rank_order_map[card_two.rank]

            # special case, can ignore
            if card_one.rank == Rank.five and card_two.rank == Rank.ace or \
                    card_one.rank == Rank.six and card_two.rank == Rank.two:
                continue

            if card_one_order+1 != card_two_order:
                is_straight = False
                break

        return is_straight

    def current_observation(self, player_number):
        num_card_per_player = []
        while True:
            index = player_number + 1;
            if index >= len(self.player_hands):
                index = 0;

            num_card_per_player.append(len(self.player_hands[index]))

            if len(num_card_per_player) == len(self.player_hands) - 1:
                break;

        return {
            "num_card_per_player": num_card_per_player,
            "cards_played": self.state,
            "your_hands": self.player_hands[player_number]
        }

    def step(self, action):
        return

    def display_all_player_hands(self):
        for idx, hand in enumerate(self.player_hands):
            print(idx, hand)

    def get_next_player(self):
        if self.current_player is None:
            for idx, player_hand in enumerate(self.player_hands):
                if have_diamond_three(player_hand[1]):
                    self.current_player = idx
                    return player_hand[0], idx

            raise ValueError("One player should have Diamond three in their hand")

        return self.player_hands[self.current_player][0], self.current_player





