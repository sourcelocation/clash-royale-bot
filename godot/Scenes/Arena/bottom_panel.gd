extends Panel

signal card_pressed(i: int)

@onready var progress_bar: ProgressBar = $ProgressBar
@onready var card_battle_scene = preload("res://Scenes/Cards/card_battle.tscn")
@onready var cards_container: HBoxContainer = $Cards

var selected_card = 0.0

func _ready() -> void:
    for c in cards_container.get_children(): c.queue_free()

func update(elixir):
    progress_bar.value = elixir

func create_cards(cards: Array):
    for c in cards_container.get_children(): c.queue_free()
    var i = 0
    for card in cards:
        var a = card_battle_scene.instantiate()
        cards_container.add_child(a)
        a.update(card)
        a.pressed.connect(card_selected.bind(i))
        i += 1

func card_selected(i):
    card_pressed.emit(i)
