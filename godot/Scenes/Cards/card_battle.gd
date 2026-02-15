extends Button
@onready var preview: TextureRect = %Preview
@onready var cost: Label = $Coin/Cost

func update(card: Card):
    preview.texture = card.texture
    cost.text = str(card.cost)
