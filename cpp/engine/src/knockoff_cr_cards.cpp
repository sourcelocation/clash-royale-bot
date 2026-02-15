#include "knockoff_cr/backend.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace knockoff_cr {

std::vector<CardDef> build_cards() {
    std::vector<CardDef> cards;
    cards.resize(kUsableCardCount);

    cards[kCardIceSpirit] = CardDef{
        "Ice Spirit", CARD_TROOP, TARGET_BOTH, 1, 1,
        3.348107244, 330, 159, 5, 0.1, 0.8, 2.5, 30.0,
    };
    cards[kCardMusketeer] = CardDef{
        "Musketeer", CARD_TROOP, TARGET_BOTH, 4, 1,
        1.674053622, 1050, 315, 5, 0.8, 0.2, 6.0, 30.0,
    };
    cards[kCardCannon] = CardDef{
        "Cannon", CARD_BUILDING, TARGET_BOTH, 3, 1,
        0.0, 1200, 310, 5, 0.95, 0.05, 5.5, 30.0,
    };
    cards[kCardHog] = CardDef{
        "Hog", CARD_TROOP, TARGET_BUILDINGS, 4, 1,
        3.348107244, 2450, 460, 5, 0.6, 1.0, 0.8, 30.0,
    };
    cards[kCardSkeletons] = CardDef{
        "Skeletons", CARD_TROOP, TARGET_GROUND, 1, 3,
        2.092567028, 100, 120, 5, 0.5, 0.5, 0.5, 30.0,
    };
    cards[kCardFireball] = CardDef{
        "Fireball", CARD_SPELL, TARGET_BOTH, 4, 1,
        0.0, 1, 1000, 300, 0.0, 0.8, 2.5, 30.0,
    };
    cards[kCardIceGolem] = CardDef{
        "Ice Golem", CARD_TROOP, TARGET_BUILDINGS, 2, 1,
        1.155096999, 1740, 122, 5, 1.0, 1.5, 0.75, 30.0,
    };
    cards[kCardLog] = CardDef{
        "Log", CARD_SPELL, TARGET_GROUND, 2, 1,
        5.022160866, 1, 380, 60, 0.0, 0.8, 1.0, 30.0,
    };

    for (auto& card : cards) {
        if (card.type == CARD_TROOP) {
            card.tower_damage = card.damage;
        }
    }

    return cards;
}

double clampd(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
}

bool is_finite(double v) {
    return std::isfinite(v) != 0;
}

double dist2(double ax, double ay, double bx, double by) {
    const double dx = ax - bx;
    const double dy = ay - by;
    return dx * dx + dy * dy;
}

} // namespace knockoff_cr
