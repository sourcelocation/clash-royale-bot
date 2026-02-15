from __future__ import annotations

from .conftest import branch_sizes, split_mask


def test_action_mask_branch_semantics(env):
    spec = env.spec()
    sizes = branch_sizes(spec)

    transition = env.reset(seed=77, options={})
    for _ in range(120):
        for team in range(2):
            mask_parts = split_mask(transition["action_mask"][team], sizes)

            wait_branch = mask_parts["wait"]
            card_branch = mask_parts["card_selection"]
            region_branch = mask_parts["position_region"]
            cell_branch = mask_parts["position_cell"]

            assert len(wait_branch) == 2
            assert float(wait_branch[1]) == 1.0

            has_playable = any(float(v) > 0.0 for v in card_branch)
            assert float(wait_branch[0]) == (1.0 if has_playable else 0.0)

            expected_fill = 1.0 if has_playable else 0.0
            assert all(float(v) == expected_fill for v in region_branch)
            assert all(float(v) == expected_fill for v in cell_branch)

            pos_masks = transition["obs"][team]["position_masks_for_all_cards"]
            for card_id, card_allowed in enumerate(card_branch):
                has_pos = any(float(v) > 0.0 for v in pos_masks[card_id])
                if has_pos:
                    assert float(card_allowed) > 0.0

        actions = [
            {"wait": 1, "card_selection": 0, "position_region": 0, "position_cell": 0},
            {"wait": 1, "card_selection": 0, "position_region": 0, "position_cell": 0},
        ]
        transition = env.step(actions)
        if transition["done"] or transition["truncation"]:
            transition = env.reset(seed=78, options={})
