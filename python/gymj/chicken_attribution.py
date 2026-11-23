"""Audit-only action attribution. No reward shaping or PPO changes here."""

from dataclasses import dataclass
from collections import defaultdict, deque

from . import _gymj


@dataclass(frozen=True)
class SubmittedDecision:
    decision_index: int
    action: object


def aggregate_chicken_attributions(settlement, history, decisions):
    """Return {(absolute_player, decision_index): signed_arena_points}.

    Supply every externally submitted decision, including forced actions and
    opponents. Applied response events may occur only after all responses arrive;
    neither their sequence numbers nor automatic draw events are decision indices.
    """
    allowed = {_gymj.ActionType.DISCARD, _gymj.ActionType.PON}

    def key(action):
        return (action.actor, int(action.type), action.tile.id, action.target)

    submitted = defaultdict(deque)
    seen = set()
    previous = -1
    for decision in decisions:
        if decision.decision_index <= previous:
            raise ValueError("decision indices must be strictly increasing")
        previous = decision.decision_index
        if decision.action.type in allowed:
            submitted[key(decision.action)].append(decision.decision_index)

    event_decisions = {}
    for sequence, event in enumerate(history):
        if event.sequence != sequence:
            raise ValueError("non-contiguous event history")
        if not event.applied or event.action.type not in allowed:
            continue
        candidates = submitted[key(event.action)]
        if not candidates:
            raise ValueError("applied action has no submitted decision")
        event_decisions[sequence] = candidates.popleft()

    totals = defaultdict(int)
    attributed = [0] * 4
    allocations = defaultdict(int)
    for entry in settlement.chicken_attributions:
        if (not 0 <= entry.player_id < 4 or
                not 0 <= entry.event_sequence < len(history) or
                not 0 <= entry.transfer_index < len(settlement.ledger.transfers)):
            raise ValueError("invalid attribution reference")
        identity = (entry.transfer_index, entry.player_id, entry.event_sequence, int(entry.kind))
        if identity in seen:
            raise ValueError("duplicate chicken attribution")
        seen.add(identity)
        event = history[entry.event_sequence]
        transfer = settlement.ledger.transfers[entry.transfer_index]
        if event.action.actor != entry.player_id or not event.applied:
            raise ValueError("attribution player does not own event")
        if event.action.tile.id != entry.tile.id or entry.points == 0:
            raise ValueError("invalid attribution tile or zero amount")
        if entry.event_sequence not in event_decisions:
            raise ValueError("attribution is not a discard or pon decision")
        if transfer.category not in (_gymj.ScoreCategory.CHICKEN,
                                     _gymj.ScoreCategory.RESPONSIBILITY_CHICKEN):
            raise ValueError("attribution points to non-chicken transfer")
        expected_player = transfer.to_player if entry.points > 0 else transfer.from_player
        if expected_player != entry.player_id:
            raise ValueError("attribution sign does not match ledger")
        allocation = (entry.transfer_index, entry.player_id)
        allocations[allocation] += abs(entry.points)
        if allocations[allocation] > transfer.points:
            raise ValueError("attribution exceeds transfer")
        decision_index = event_decisions[entry.event_sequence]
        totals[(entry.player_id, decision_index)] += entry.points
        attributed[entry.player_id] += entry.points

    actual = [0] * 4
    for transfer in settlement.ledger.transfers:
        if transfer.category in (_gymj.ScoreCategory.CHICKEN,
                                 _gymj.ScoreCategory.RESPONSIBILITY_CHICKEN):
            actual[transfer.from_player] -= transfer.points
            actual[transfer.to_player] += transfer.points
    if any(attributed[p] + settlement.chicken_unattributed_deltas[p] != actual[p]
           for p in range(4)):
        raise ValueError("chicken attribution does not reconcile with ledger")
    return dict(totals)
