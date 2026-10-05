"""Strict public decisions for the host-controlled Astra tools."""
from __future__ import annotations

import copy
import json
import math

SCHEMA_VERSION = 'astra-libero-v1'
IDS = ('episode_id', 'intervention_id', 'request_id', 'observation_id', 'proposal_id')
MAX_JSON_BYTES = 65536
MAX_TARGET_POSITION_NORM_M = 0.05
MAX_TARGET_ROTATION_NORM_RAD = 0.35
MIN_EEF_CHUNK_ACTIONS = 30
MAX_EEF_CHUNK_ACTIONS = 50


class ProtocolError(ValueError):
    """A rejected input that must not have caused execution."""


def strict_json_loads(raw):
    if not isinstance(raw, str) or len(raw.encode('utf-8')) > MAX_JSON_BYTES:
        raise ProtocolError('invalid_json_size')

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ProtocolError('duplicate_json_key')
            result[key] = value
        return result

    def constant(_value):
        raise ProtocolError('nonfinite_json_number')

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, RecursionError) as error:
        raise ProtocolError('invalid_json') from error


def exact_keys(value, expected, name):
    if type(value) is not dict or set(value) != set(expected):
        raise ProtocolError(name + ':wrong_keys')


def text(value, name, limit=4096):
    if type(value) is not str or not value.strip() or len(value) > limit:
        raise ProtocolError(name + ':invalid_text')


def enum(value, choices, name):
    if type(value) is not str or value not in choices:
        raise ProtocolError(name + ':invalid_enum')


def vector(value, size, name):
    if type(value) is not list or len(value) != size:
        raise ProtocolError(name + ':wrong_shape')
    try:
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in value):
            raise ProtocolError(name + ':invalid_number')
        return [float(x) for x in value]
    except (OverflowError, ValueError) as error:
        raise ProtocolError(name + ':invalid_number') from error


def _common_keys():
    return (*IDS, 'schema_version', 'decision_id', 'mode', 'reason')


def _validate_chunk_edits(edits):
    exact_keys(edits, ('execute_steps', 'waypoint_edits'), 'chunk_edits')
    if type(edits['execute_steps']) is not int or not 1 <= edits['execute_steps'] <= 50:
        raise ProtocolError('chunk_edit_execute_steps_invalid')
    rows = edits['waypoint_edits']
    if type(rows) is not list or not 1 <= len(rows) <= 50:
        raise ProtocolError('chunk_edit_waypoints_invalid')
    seen_steps = set()
    for row in rows:
        exact_keys(row, ('proposal_step', 'delta_position_m',
                         'delta_rotation_vector_rad', 'gripper'), 'waypoint_edit')
        if type(row['proposal_step']) is not int or not 0 <= row['proposal_step'] < 50:
            raise ProtocolError('chunk_edit_proposal_step_invalid')
        if row['proposal_step'] in seen_steps:
            raise ProtocolError('chunk_edit_duplicate_proposal_step')
        seen_steps.add(row['proposal_step'])
        position_delta = vector(row['delta_position_m'], 3, 'delta_position_m')
        rotation_delta = vector(row['delta_rotation_vector_rad'], 3,
                                'delta_rotation_vector_rad')
        if math.sqrt(sum(x * x for x in position_delta)) > MAX_TARGET_POSITION_NORM_M + 1e-9:
            raise ProtocolError('chunk_edit_position_delta_too_large')
        if math.sqrt(sum(x * x for x in rotation_delta)) > MAX_TARGET_ROTATION_NORM_RAD + 1e-9:
            raise ProtocolError('chunk_edit_rotation_delta_too_large')
        row['delta_position_m'] = position_delta
        row['delta_rotation_vector_rad'] = rotation_delta
        enum(row['gripper'], ('keep', 'open', 'closed'), 'chunk_edit_gripper')


def _validate_eef_chunk(chunk):
    """Validate an Astra-owned sequence of one-step EEF corrections."""
    exact_keys(chunk, ('actions',), 'eef_chunk')
    actions = chunk['actions']
    if (type(actions) is not list or
            not MIN_EEF_CHUNK_ACTIONS <= len(actions) <= MAX_EEF_CHUNK_ACTIONS):
        raise ProtocolError('eef_chunk_action_count_invalid')
    for action in actions:
        exact_keys(action, ('delta_position', 'delta_rotation_vector', 'gripper'),
                   'eef_chunk_action')
        position = vector(action['delta_position'], 3, 'chunk_delta_position')
        rotation = vector(action['delta_rotation_vector'], 3, 'chunk_delta_rotation_vector')
        if math.sqrt(sum(x * x for x in position)) > MAX_TARGET_POSITION_NORM_M + 1e-9:
            raise ProtocolError('chunk_delta_exceeds_total_position_bound')
        if math.sqrt(sum(x * x for x in rotation)) > MAX_TARGET_ROTATION_NORM_RAD + 1e-9:
            raise ProtocolError('chunk_delta_exceeds_total_rotation_bound')
        enum(action['gripper'], ('keep', 'open', 'closed'), 'chunk_gripper')
        action['delta_position'] = position
        action['delta_rotation_vector'] = rotation


def validate_decision(value, expected_ids, *, expected_modes=None):
    """Validate a compact model decision and add host-owned fixed fields.

    ``steps``, ``target``, ``assessment``, and the unused action payload are
    deliberately not model parameters. The host derives fixed steps and null
    fields from the selected tool/mode.
    """
    if type(value) is not dict:
        raise ProtocolError('decision:wrong_keys')
    d = copy.deepcopy(value)
    mode = d.get('mode')
    if mode == 'eef_delta':
        expected = (*_common_keys(), 'delta')
    elif mode == 'eef_chunk':
        expected = (*_common_keys(), 'chunk')
    elif mode == 'edit_pi05_chunk':
        expected = (*_common_keys(), 'chunk_edits')
    elif mode in ('resume_pi05', 'stop'):
        expected = _common_keys()
    else:
        raise ProtocolError('mode:invalid_enum')
    exact_keys(d, expected, 'decision')
    if expected_modes is not None and mode not in expected_modes:
        raise ProtocolError('wrong_tool_for_decision_mode')
    if d['schema_version'] != SCHEMA_VERSION:
        raise ProtocolError('wrong_schema_version')
    for key in IDS:
        text(d[key], key, 160)
        if d[key] != expected_ids[key]:
            raise ProtocolError('stale_or_wrong_' + key)
    text(d['decision_id'], 'decision_id', 160)
    text(d['reason'], 'reason')

    raw_chunk_edits = d.get('chunk_edits')
    raw_eef_chunk = d.get('chunk')
    d['target'] = None
    d['chunk_edits'] = None
    d['chunk'] = None
    if mode in ('resume_pi05', 'stop'):
        d['steps'] = 0
        d['delta'] = None
        return d
    if mode == 'edit_pi05_chunk':
        d['steps'] = 0
        d['delta'] = None
        _validate_chunk_edits(raw_chunk_edits)
        d['chunk_edits'] = raw_chunk_edits
        return d
    if mode == 'eef_chunk':
        _validate_eef_chunk(raw_eef_chunk)
        d['steps'] = len(raw_eef_chunk['actions'])
        d['delta'] = None
        d['chunk'] = raw_eef_chunk
        return d

    d['steps'] = 1
    target = d['delta']
    exact_keys(target, ('delta_position', 'delta_rotation_vector', 'gripper'), 'delta')
    for key in ('delta_position', 'delta_rotation_vector'):
        target[key] = vector(target[key], 3, key)
    enum(target['gripper'], ('keep', 'open', 'closed'), 'gripper')
    if math.sqrt(sum(x * x for x in target['delta_position'])) > MAX_TARGET_POSITION_NORM_M + 1e-9:
        raise ProtocolError('delta_exceeds_total_position_bound')
    if math.sqrt(sum(x * x for x in target['delta_rotation_vector'])) > MAX_TARGET_ROTATION_NORM_RAD + 1e-9:
        raise ProtocolError('delta_exceeds_total_rotation_bound')
    return d


def _obj(properties, required=None):
    return dict(type='object', properties=properties,
                required=list(properties if required is None else required),
                additionalProperties=False)


def decision_schema(modes=None):
    if modes is None:
        modes = ['eef_delta', 'resume_pi05', 'stop']
    modes = list(modes)
    identifier = dict(type='string', minLength=1, maxLength=160)
    string = dict(type='string', minLength=1, maxLength=4096)
    v3 = dict(type='array', minItems=3, maxItems=3, items=dict(type='number'))
    bounded_position_delta = dict(
        type='array', minItems=3, maxItems=3,
        description='Position increments [dx, dy, dz] in meters relative to the measured EEF pose.',
        items=dict(type='number', minimum=-MAX_TARGET_POSITION_NORM_M,
                   maximum=MAX_TARGET_POSITION_NORM_M),
    )
    bounded_rotation_delta = dict(
        type='array', minItems=3, maxItems=3,
        description='World-frame rotation increments [drx, dry, drz] in radians.',
        items=dict(type='number', minimum=-MAX_TARGET_ROTATION_NORM_RAD,
                   maximum=MAX_TARGET_ROTATION_NORM_RAD),
    )
    chunk_edits = _obj(dict(
        execute_steps=dict(type='integer', minimum=1, maximum=50),
        waypoint_edits=dict(type='array', minItems=1, maxItems=50, items=_obj(dict(
            proposal_step=dict(type='integer', minimum=0, maximum=49),
            delta_position_m=v3,
            delta_rotation_vector_rad=v3,
            gripper=dict(type='string', enum=['keep', 'open', 'closed']),
        ))),
    ))
    eef_chunk = _obj(dict(
        actions=dict(type='array', minItems=MIN_EEF_CHUNK_ACTIONS,
                     maxItems=MAX_EEF_CHUNK_ACTIONS, items=_obj(dict(
            delta_position=bounded_position_delta,
            delta_rotation_vector=bounded_rotation_delta,
            gripper=dict(type='string', enum=['keep', 'open', 'closed']),
        ))),
    ))
    properties = dict(
        schema_version=dict(type='string', enum=[SCHEMA_VERSION]),
        **{key: identifier for key in IDS},
        decision_id=identifier,
        mode=dict(type='string', enum=modes),
        reason=string,
    )
    required = list(_common_keys())
    if 'eef_delta' in modes:
        properties['delta'] = _obj(dict(
            delta_position=bounded_position_delta,
            delta_rotation_vector=bounded_rotation_delta,
            gripper=dict(type='string', enum=['keep', 'open', 'closed']),
        ))
        if len(modes) == 1:
            required.append('delta')
    if 'edit_pi05_chunk' in modes:
        properties['chunk_edits'] = chunk_edits
        if len(modes) == 1:
            required.append('chunk_edits')
    if 'eef_chunk' in modes:
        properties['chunk'] = eef_chunk
        if len(modes) == 1:
            required.append('chunk')
    return _obj(properties, required)


def tool_specs():
    identifier = dict(type='string', minLength=1, maxLength=160)
    return [
        dict(type='function', name='libero_observe',
             description='Read synchronized public RGB and robot state; does not step.',
             inputSchema=_obj(dict(episode_id=identifier, intervention_id=identifier))),
        dict(type='function', name='pi05_propose',
             description='Read a fresh unexecuted pi05 proposal; does not step or update CALL.',
             inputSchema=_obj(dict(request_id=identifier, observation_id=identifier))),
        dict(type='function', name='libero_execute_eef',
             description=('Execute exactly one in-range dimensionful EEF delta. The host fixes steps=1; '
                          'provide only delta_position, delta_rotation_vector, and gripper.'),
             inputSchema=decision_schema(['eef_delta'])),
        dict(type='function', name='libero_execute_eef_chunk',
             description=(
                 'Execute an Astra-generated sequence of 30 to 50 one-step EEF corrections. '
                 'The host executes the complete sequence without an intermediate model '
                 'observation or Pi0.5 proposal, then refreshes both.'),
             inputSchema=decision_schema(['eef_chunk'])),
        dict(type='function', name='libero_edit_pi05_chunk',
             description=('Optional alternative: edit the current unexecuted Pi0.5 nominal trajectory '
                          'with sparse bounded waypoint offsets, then execute its prefix.'),
             inputSchema=decision_schema(['edit_pi05_chunk'])),
        dict(type='function', name='libero_resume_pi05',
             description='Hand control back; the host verifies the exact fresh checkpoint.',
             inputSchema=decision_schema(['resume_pi05'])),
    ]
