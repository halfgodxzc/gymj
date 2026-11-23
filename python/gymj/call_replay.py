"""Screen and replay Pon/Kan alternatives without treating interventions as PPO."""
from __future__ import annotations

import hashlib
import math
from types import SimpleNamespace
import numpy as np
import torch

from ._gymj import Arena, ActionEncoder, DecisionFeatureEncoder, TILE_TYPE_COUNT, ACTION_SPACE_SIZE, calculate_shanten
from .teacher import _own_meld_count, PON_ACTION_ID, PASS_ACTION_ID, _legal_kan_action_ids
from .v4_features import tensorize_v4_decision
from .collect_q_replays import root_inputs, arrays_fingerprint
from .diagnostic_dataset import _terminal_record

CALL_REPLAY_VERSION = 1


def pon_screen(hand, meld_count, called_tile, discard_mask):
    if len(hand)!=TILE_TYPE_COUNT or len(discard_mask)!=TILE_TYPE_COUNT:
        raise ValueError('Invalid tile vector size')
    if not 0<=called_tile<TILE_TYPE_COUNT or hand[called_tile]<2 or not 0<=meld_count<4:
        raise ValueError('Invalid Pon hand')
    before=int(calculate_shanten(hand,meld_count).best)
    after=list(hand)
    after[called_tile]-=2
    scores={}
    for tile,legal in enumerate(discard_mask):
        if not legal:
            continue
        if after[tile]<=0:
            raise ValueError('Follow-up discard absent from hand')
        trial=after.copy()
        trial[tile]-=1
        scores[tile]=int(calculate_shanten(trial,meld_count+1).best)
    if not scores:
        raise ValueError('Pon has no legal follow-up discard')
    best=min(scores.values())
    return dict(before_shanten=before,best_after_shanten=best,eligible=best<=before,
                best_discard_ids=[t for t,s in scores.items() if s==best],discard_shanten=scores)


def alternative_pairs(legal_ids, selected, pon=None, logits=None):
    legal=set(legal_ids)
    if selected not in legal:
        raise ValueError('Recorded action is illegal')
    pairs=[]
    if {PON_ACTION_ID,PASS_ACTION_ID}<=legal and selected in (PON_ACTION_ID,PASS_ACTION_ID):
        if pon is None:
            raise ValueError('Missing Pon assessment')
        if pon['eligible']:
            pairs.append(dict(family='pon',original=selected,
                alternative=PASS_ACTION_ID if selected==PON_ACTION_ID else PON_ACTION_ID,screen=pon))
    kans=set(_legal_kan_action_ids(tuple(sorted(legal))))
    if selected in kans:
        nonkan=sorted(legal-kans)
        if nonkan:
            if logits is None or any(not math.isfinite(float(logits[a])) for a in nonkan):
                raise ValueError('Missing finite non-Kan scores')
            fallback=max(nonkan,key=lambda a:(float(logits[a]),-a))
            pairs.append(dict(family='kan',original=selected,alternative=fallback,kan_action=selected,
                              nonkan_rule='highest_probability_nonkan'))
    else:
        for kan in sorted(kans):
            pairs.append(dict(family='kan',original=selected,alternative=kan,kan_action=kan,
                              nonkan_rule='recorded_nonkan_action'))
    return pairs


def new_arena(episode):
    if len(episode['seats'])!=4:
        raise ValueError('Expected four seat policies')
    arena=Arena(seed=episode['seed'],dealer=episode['dealer'])
    arena.start_round()
    return arena


def submit_record(arena, record):
    if arena.finished:
        raise ValueError('Recorded prefix exceeds round')
    player,action_id=map(int,record)
    decision=arena.current_decision()
    if int(decision.player)!=player:
        raise ValueError('Replay player mismatch')
    encoded=ActionEncoder.encode_legal_actions(decision.legal_actions)
    if not encoded.is_legal(action_id):
        raise ValueError('Replay action mismatch')
    arena.submit_action(encoded.action_at(action_id))


def fingerprint(arena):
    state=SimpleNamespace(player_id=int(arena.current_decision().player))
    return arrays_fingerprint(root_inputs(SimpleNamespace(_arena=arena),state))


def restore(episode,case):
    index=case['decision_index']
    if not 0<=index<len(episode['actions']):
        raise ValueError('Invalid root decision index')
    if case['episode_seed']!=episode['seed'] or case['seats']!=list(episode['seats']):
        raise ValueError('Root episode identity mismatch')
    if episode['actions'][index]!=[case['player_id'],case['original']]:
        raise ValueError('Root original action mismatch')
    arena=new_arena(episode)
    for record in episode['actions'][:index]:
        submit_record(arena,record)
    if fingerprint(arena)!=case['root_sha256']:
        raise ValueError('Root state mismatch')
    return arena


def policy_logits(agent,features):
    with torch.inference_mode():
        return (agent.actor(tensorize_v4_decision(features,device=agent.device))[0]/agent.temperature).cpu().numpy()


def screen_episode(episode,agents):
    arena=new_arena(episode)
    cases=[]
    for step,record in enumerate(episode['actions']):
        d=arena.current_decision()
        player,selected=map(int,record)
        if player!=int(d.player):
            raise ValueError('Screen player mismatch')
        enc=ActionEncoder.encode_legal_actions(d.legal_actions)
        legal=[a for a in range(ACTION_SPACE_SIZE) if enc.is_legal(a)]
        kans=_legal_kan_action_ids(tuple(legal))
        want_pon=PON_ACTION_ID in legal and PASS_ACTION_ID in legal and selected in (PON_ACTION_ID,PASS_ACTION_ID)
        if want_pon or kans:
            features=DecisionFeatureEncoder.encode(arena,player)
            assessment=None
            if want_pon:
                follow=features.pon_follow_up
                if not follow.available:
                    raise ValueError('Legal Pon missing follow-up features')
                obs=features.observation
                assessment=pon_screen(list(obs.hand_counts),_own_meld_count(obs),int(obs.current_discard),follow.discard_mask)
            logits=policy_logits(agents[episode['seats'][player]],features) if selected in kans else None
            pairs=alternative_pairs(legal,selected,assessment,logits)
            if pairs:
                sha=fingerprint(arena)
                for pair in pairs:
                    cases.append(dict(**pair,episode_seed=episode['seed'],dealer=episode['dealer'],
                        decision_index=step,player_id=player,seats=list(episode['seats']),legal_ids=legal,root_sha256=sha))
        submit_record(arena,record)
    if not arena.finished or list(arena.round_settlement.ledger.score_deltas)!=episode['scores']:
        raise ValueError('Source episode did not replay exactly')
    return cases


def terminal(arena):
    record=_terminal_record(SimpleNamespace(_arena=arena,round_result=arena.result,round_settlement=arena.round_settlement))
    if sum(record['score_deltas'])!=0:
        raise ValueError('Nonzero terminal ledger')
    return record


def repetition_seed(case,repeat,policy_seed):
    key=f"call-v1:{policy_seed}:{case['episode_seed']}:{case['decision_index']}:{case['original']}:{case['alternative']}:{repeat}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')


def continue_branch(episode,case,action_id,agents,repeat,policy_seed,max_steps=1000):
    if action_id not in (case['original'],case['alternative']):
        raise ValueError('Unselected intervention')
    if max_steps<1 or repeat<0:
        raise ValueError('Invalid continuation budget')
    arena=restore(episode,case)
    seed=repetition_seed(case,repeat,policy_seed)
    rngs=[np.random.default_rng(np.random.SeedSequence([seed,p])) for p in range(4)]
    actions=[]
    for offset in range(max_steps):
        d=arena.current_decision()
        seat=int(d.player)
        model=episode['seats'][seat]
        enc=ActionEncoder.encode_legal_actions(d.legal_actions)
        if offset==0:
            chosen,logp,behavior=action_id,None,'intervention'
        else:
            logits=policy_logits(agents[model],DecisionFeatureEncoder.encode(arena,seat))
            valid=np.isfinite(logits)
            if not valid.any():
                raise ValueError('No selectable action')
            probabilities=np.zeros(len(logits),dtype=np.float64)
            probabilities[valid]=np.exp(logits[valid].astype(np.float64)-float(logits[valid].max()))
            probabilities/=probabilities.sum()
            chosen=int(rngs[seat].choice(len(logits),p=probabilities))
            logp=float(np.log(probabilities[chosen]))
            behavior='policy_sample' if valid.sum()>1 else 'forced_legal'
        if not enc.is_legal(chosen):
            raise ValueError('Illegal branch action')
        action=enc.action_at(chosen)
        actions.append(dict(player_id=seat,policy_index=model,action_id=chosen,behavior=behavior,
            behavior_log_probability=logp,type=action.type.name,tile=int(action.tile.id),target=int(action.target)))
        arena.submit_action(action)
        if arena.finished:
            return dict(action_id=action_id,repeat=repeat,sampling_seed=seed,root_sha256=case['root_sha256'],
                actions=actions,terminal=terminal(arena))
    raise RuntimeError('Continuation action limit exceeded')


def audit_branch(episode,case,branch):
    if branch['root_sha256']!=case['root_sha256'] or branch['action_id'] not in (case['original'],case['alternative']):
        raise ValueError('Branch identity mismatch')
    arena=restore(episode,case)
    for offset,row in enumerate(branch['actions']):
        if row['policy_index']!=episode['seats'][row['player_id']]:
            raise ValueError('Branch policy mismatch')
        if offset==0 and (row['action_id']!=branch['action_id'] or row['behavior']!='intervention' or row['behavior_log_probability'] is not None):
            raise ValueError('Invalid intervention record')
        submit_record(arena,[row['player_id'],row['action_id']])
    if not arena.finished or terminal(arena)!=branch['terminal']:
        raise ValueError('Branch terminal mismatch')


def summarize(case,branches,repeats):
    if repeats<2 or len(branches)!=2*repeats:
        raise ValueError('Incomplete repetitions')
    values=np.zeros((2,repeats,4),dtype=np.float64)
    seen=set()
    seeds={}
    for branch in branches:
        action=branch['action_id']
        if action not in (case['original'],case['alternative']):
            raise ValueError('Unexpected branch action')
        slot=0 if action==case['original'] else 1
        repeat=branch['repeat']
        if not 0<=repeat<repeats or (slot,repeat) in seen or branch['root_sha256']!=case['root_sha256']:
            raise ValueError('Invalid branch identity')
        seen.add((slot,repeat))
        if repeat in seeds and seeds[repeat]!=branch['sampling_seed']:
            raise ValueError('Unpaired random seeds')
        seeds[repeat]=branch['sampling_seed']
        values[slot,repeat]=branch['terminal']['score_deltas']
        if values[slot,repeat].sum()!=0:
            raise ValueError('Nonzero settlement')
    if len(set(seeds.values()))!=repeats:
        raise ValueError('Repeated continuation seed')
    player=case['player_id']
    delta=values[0,:,player]-values[1,:,player]
    return dict(repeats_per_action=repeats,original_mean=float(values[0,:,player].mean()),
        alternative_mean=float(values[1,:,player].mean()),original_minus_alternative=float(delta.mean()),
        paired_differences=delta.tolist(),paired_standard_error=float(delta.std(ddof=1)/math.sqrt(repeats)),
        four_player_means=values.mean(axis=1).tolist(),training_scope='auxiliary_pair_only_not_on_policy_ppo')
