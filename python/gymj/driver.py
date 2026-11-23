from ._gymj import ActionType, Arena, RoundPhase


def first_action(decision, action_type, tile_id=None):
    for action in decision.legal_actions:
        if action.type == action_type and (
            tile_id is None or action.tile.id == tile_id
        ):
            return action
    raise AssertionError(
        f"No legal {action_type.name} action"
        + ("" if tile_id is None else f" for tile {tile_id}")
    )


def start_round_passing_initial_ready(arena):
    arena.start_round()
    while arena.phase == RoundPhase.AWAIT_READY_DECLARATION:
        decision = arena.current_decision()
        arena.submit_action(first_action(decision, ActionType.PASS))


def pass_all_responses(arena):
    if arena.phase == RoundPhase.AWAIT_READY_DECLARATION:
        decision = arena.current_decision()
        arena.submit_action(first_action(decision, ActionType.PASS))
    for _ in range(3):
        decision = arena.current_decision()
        arena.submit_action(first_action(decision, ActionType.PASS))
    if arena.phase == RoundPhase.AWAIT_READY_DECLARATION:
        decision = arena.current_decision()
        arena.submit_action(first_action(decision, ActionType.PASS))


def run_passive_game(seed):
    arena = Arena(seed=seed)
    arena.start_round()

    while not arena.finished:
        decision = arena.current_decision()
        if decision.phase in (
            RoundPhase.AWAIT_READY_DECLARATION,
            RoundPhase.AWAIT_DISCARD_RESPONSES,
            RoundPhase.AWAIT_KAKAN_RESPONSES,
        ):
            action = first_action(decision, ActionType.PASS)
        else:
            action = first_action(decision, ActionType.DISCARD)
        arena.submit_action(action)

    return arena
