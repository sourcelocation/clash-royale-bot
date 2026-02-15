extends Resource
class_name CardsRes

@export var cards: Array[Card] = []

func get_card_by_name(card_name: String) -> Card:
	for card in cards:
		if card.name == card_name:
			return card
	return null

func get_cards_by_type(card_type: Card.CardType) -> Array[Card]:
	var filtered_cards: Array[Card] = []
	for card in cards:
		if card.type == card_type:
			filtered_cards.append(card)
	return filtered_cards

func get_card_count() -> int:
	return cards.size()
