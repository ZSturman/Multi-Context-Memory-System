from __future__ import annotations

from copy import deepcopy


def _tick(
    timestamp: float,
    *,
    location: str,
    goal: str,
    thought: str,
    vision: dict,
    audio: dict,
    time_context: dict,
    emotion: dict,
) -> dict:
    return {
        "timestamp": timestamp,
        "location": location,
        "goal_state": goal,
        "thought": thought,
        "modalities": {
            "vision": vision,
            "audio": audio,
            "time_context": time_context,
            "emotion": emotion,
        },
    }


def scenario_library() -> dict[str, list[dict]]:
    base_time = {"feature_vector": {"hour_night": 1.0, "context_shift": 0.0}, "hypothesis": "nighttime_kitchen", "context_labels": ["time:night", "task:cooking", "kitchen"]}
    calm_emotion = {"feature_vector": {"arousal": 0.10, "valence": 0.10}, "hypothesis": "calm", "context_labels": ["emotion:calm"], "emotion_hint": "calm"}
    startled_emotion = {"feature_vector": {"arousal": 0.92, "valence": -0.35}, "hypothesis": "startled", "context_labels": ["emotion:startled"], "emotion_hint": "startled"}
    resolved_emotion = {"feature_vector": {"arousal": 0.28, "valence": 0.02}, "hypothesis": "settling", "context_labels": ["emotion:settling"], "emotion_hint": "settling"}

    surprise_then_resolution = [
        _tick(
            0.0,
            location="kitchen",
            goal="cook_dinner",
            thought="monitor_environment",
            vision={"feature_vector": {"scene_stable": 1.0, "pan_visible": 0.2}, "hypothesis": "kitchen_idle", "context_labels": ["kitchen"]},
            audio={"feature_vector": {"ambient_hum": 0.2}, "hypothesis": "quiet_room", "context_labels": ["kitchen"]},
            time_context=deepcopy(base_time),
            emotion=deepcopy(calm_emotion),
        ),
        _tick(
            1.0,
            location="kitchen",
            goal="cook_dinner",
            thought="what_was_that",
            vision={"feature_vector": {"scene_stable": 0.3, "object_motion": 0.4}, "hypothesis": "motion_left", "context_labels": ["kitchen"]},
            audio={"feature_vector": {"bang_intensity": 1.0, "metallic": 0.8, "impact": 1.0, "intensity": 1.0}, "hypothesis": "metallic_bang", "candidate_category": "cookware_fall", "context_labels": ["kitchen"], "consequence": 0.8},
            time_context={"feature_vector": {"hour_night": 1.0, "context_shift": 0.4}, "hypothesis": "nighttime_interruption", "context_labels": ["time:night", "task:interrupted", "kitchen"]},
            emotion=deepcopy(startled_emotion),
        ),
        _tick(
            2.0,
            location="kitchen",
            goal="inspect_noise",
            thought="check_left_side",
            vision={"feature_vector": {"scene_stable": 0.1, "pan_on_floor": 1.0, "object_motion": 0.7}, "hypothesis": "pan_on_floor", "candidate_category": "fallen_pan", "context_labels": ["kitchen", "floor"]},
            audio={"feature_vector": {"bang_intensity": 0.5, "metallic": 0.7, "impact": 0.6, "intensity": 0.5}, "hypothesis": "metallic_resonance", "context_labels": ["kitchen"]},
            time_context={"feature_vector": {"hour_night": 1.0, "context_shift": 0.2}, "hypothesis": "inspection_phase", "context_labels": ["time:night", "task:inspection", "kitchen"]},
            emotion=deepcopy(startled_emotion),
        ),
        _tick(
            3.0,
            location="kitchen",
            goal="resume_cooking",
            thought="it_was_the_pan",
            vision={"feature_vector": {"scene_stable": 0.8, "pan_on_floor": 0.9}, "hypothesis": "pan_confirmed", "candidate_category": "fallen_pan", "context_labels": ["kitchen", "floor"]},
            audio={"feature_vector": {"ambient_hum": 0.1, "intensity": 0.1}, "hypothesis": "quiet_after_bang", "context_labels": ["kitchen"]},
            time_context={"feature_vector": {"hour_night": 1.0, "context_shift": 0.1}, "hypothesis": "resolution_phase", "context_labels": ["time:night", "task:resolving", "kitchen"]},
            emotion=deepcopy(resolved_emotion),
        ),
        _tick(
            4.0,
            location="kitchen",
            goal="resume_cooking",
            thought="resolved",
            vision={"feature_vector": {"scene_stable": 1.0, "pan_on_floor": 0.8}, "hypothesis": "kitchen_stable_after_pan", "context_labels": ["kitchen"]},
            audio={"feature_vector": {"ambient_hum": 0.2, "intensity": 0.0}, "hypothesis": "quiet_room", "context_labels": ["kitchen"]},
            time_context=deepcopy(base_time),
            emotion=deepcopy(resolved_emotion),
        ),
    ]

    same_cue_later = [
        _tick(
            10.0,
            location="kitchen",
            goal="cook_dinner",
            thought="monitor_environment",
            vision={"feature_vector": {"scene_stable": 1.0, "pan_visible": 0.4}, "hypothesis": "kitchen_idle", "context_labels": ["kitchen"]},
            audio={"feature_vector": {"ambient_hum": 0.2}, "hypothesis": "quiet_room", "context_labels": ["kitchen"]},
            time_context=deepcopy(base_time),
            emotion=deepcopy(calm_emotion),
        ),
        _tick(
            11.0,
            location="kitchen",
            goal="cook_dinner",
            thought="possible_repeat",
            vision={"feature_vector": {"scene_stable": 0.4, "object_motion": 0.3}, "hypothesis": "motion_left", "context_labels": ["kitchen"]},
            audio={"feature_vector": {"bang_intensity": 0.95, "metallic": 0.78, "impact": 0.95, "intensity": 0.95}, "hypothesis": "metallic_bang", "candidate_category": "cookware_fall", "context_labels": ["kitchen"], "consequence": 0.7},
            time_context={"feature_vector": {"hour_night": 1.0, "context_shift": 0.35}, "hypothesis": "nighttime_interruption", "context_labels": ["time:night", "task:interrupted", "kitchen"]},
            emotion=deepcopy(startled_emotion),
        ),
        _tick(
            12.0,
            location="kitchen",
            goal="inspect_noise",
            thought="check_pan_again",
            vision={"feature_vector": {"scene_stable": 0.2, "pan_on_floor": 0.9}, "hypothesis": "pan_on_floor", "candidate_category": "fallen_pan", "context_labels": ["kitchen", "floor"]},
            audio={"feature_vector": {"bang_intensity": 0.2, "metallic": 0.2, "intensity": 0.2}, "hypothesis": "quiet_after_bang", "context_labels": ["kitchen"]},
            time_context={"feature_vector": {"hour_night": 1.0, "context_shift": 0.15}, "hypothesis": "inspection_phase", "context_labels": ["time:night", "task:inspection", "kitchen"]},
            emotion=deepcopy(resolved_emotion),
        ),
    ]

    false_cue_mismatch = [
        _tick(
            20.0,
            location="kitchen",
            goal="cook_dinner",
            thought="monitor_environment",
            vision={"feature_vector": {"scene_stable": 1.0}, "hypothesis": "kitchen_idle", "context_labels": ["kitchen"]},
            audio={"feature_vector": {"ambient_hum": 0.2}, "hypothesis": "quiet_room", "context_labels": ["kitchen"]},
            time_context=deepcopy(base_time),
            emotion=deepcopy(calm_emotion),
        ),
        _tick(
            21.0,
            location="kitchen",
            goal="inspect_noise",
            thought="different_cause?",
            vision={"feature_vector": {"scene_stable": 0.7, "water_drip_visible": 0.9}, "hypothesis": "leaking_faucet", "candidate_category": "water_impact", "context_labels": ["kitchen", "sink"]},
            audio={"feature_vector": {"bang_intensity": 0.88, "metallic": 0.15, "impact": 0.90, "water_like": 0.9, "intensity": 0.88}, "hypothesis": "sink_slam", "candidate_category": "water_impact", "context_labels": ["kitchen", "sink"], "consequence": 0.6},
            time_context={"feature_vector": {"hour_night": 1.0, "context_shift": 0.30}, "hypothesis": "nighttime_interruption", "context_labels": ["time:night", "task:interrupted", "kitchen"]},
            emotion=deepcopy(startled_emotion),
        ),
        _tick(
            22.0,
            location="kitchen",
            goal="fix_sink",
            thought="not_the_pan",
            vision={"feature_vector": {"scene_stable": 0.4, "water_drip_visible": 1.0}, "hypothesis": "leaking_faucet_confirmed", "candidate_category": "water_impact", "context_labels": ["kitchen", "sink"]},
            audio={"feature_vector": {"water_like": 0.7, "intensity": 0.3}, "hypothesis": "dripping_sink", "context_labels": ["kitchen", "sink"]},
            time_context={"feature_vector": {"hour_night": 1.0, "context_shift": 0.15}, "hypothesis": "inspection_phase", "context_labels": ["time:night", "task:inspection", "kitchen"]},
            emotion=deepcopy(resolved_emotion),
        ),
    ]

    intrusive_repeated_poor_fit = [
        _tick(
            30.0,
            location="hallway",
            goal="walk_to_room",
            thought="remember_noise",
            vision={"feature_vector": {"scene_stable": 0.9, "shadow_left": 0.2}, "hypothesis": "dim_hallway", "context_labels": ["hallway"]},
            audio={"feature_vector": {"soft_creak": 0.55, "impact": 0.35, "intensity": 0.45}, "hypothesis": "soft_creak", "candidate_category": "ambiguous_noise", "context_labels": ["hallway"], "consequence": 0.2},
            time_context={"feature_vector": {"hour_night": 1.0, "context_shift": 0.25}, "hypothesis": "night_walk", "context_labels": ["time:night", "hallway"]},
            emotion=deepcopy(startled_emotion),
        ),
        _tick(
            31.0,
            location="hallway",
            goal="walk_to_room",
            thought="ignore_it",
            vision={"feature_vector": {"scene_stable": 1.0, "shadow_left": 0.1}, "hypothesis": "empty_hallway", "context_labels": ["hallway"]},
            audio={"feature_vector": {"soft_creak": 0.35, "impact": 0.25, "intensity": 0.30}, "hypothesis": "house_settling", "candidate_category": "ambiguous_noise", "context_labels": ["hallway"], "consequence": 0.1},
            time_context={"feature_vector": {"hour_night": 1.0, "context_shift": 0.05}, "hypothesis": "night_walk", "context_labels": ["time:night", "hallway"]},
            emotion=deepcopy(calm_emotion),
        ),
        _tick(
            32.0,
            location="office",
            goal="work_quietly",
            thought="stop_recalling_pan",
            vision={"feature_vector": {"scene_stable": 1.0, "desk_visible": 1.0}, "hypothesis": "office_settled", "context_labels": ["office"]},
            audio={"feature_vector": {"keyboard_click": 0.30, "impact": 0.28, "intensity": 0.22}, "hypothesis": "keyboard_noise", "candidate_category": "office_noise", "context_labels": ["office"], "consequence": 0.0},
            time_context={"feature_vector": {"hour_night": 1.0, "context_shift": 0.45}, "hypothesis": "office_context", "context_labels": ["time:night", "office"]},
            emotion=deepcopy(calm_emotion),
        ),
    ]

    repeated_pattern_abstraction = surprise_then_resolution + same_cue_later + [
        _tick(
            40.0,
            location="kitchen",
            goal="clean_kitchen",
            thought="another_metallic_drop",
            vision={"feature_vector": {"scene_stable": 0.3, "utensil_on_floor": 0.9}, "hypothesis": "utensil_fell", "candidate_category": "cookware_fall", "context_labels": ["kitchen", "floor"]},
            audio={"feature_vector": {"bang_intensity": 0.9, "metallic": 0.82, "impact": 0.92, "intensity": 0.9}, "hypothesis": "metallic_bang", "candidate_category": "cookware_fall", "context_labels": ["kitchen"], "consequence": 0.75},
            time_context={"feature_vector": {"hour_night": 0.4, "context_shift": 0.30}, "hypothesis": "cleanup_interruption", "context_labels": ["time:evening", "task:cleaning", "kitchen"]},
            emotion=deepcopy(startled_emotion),
        ),
        _tick(
            41.0,
            location="kitchen",
            goal="clean_kitchen",
            thought="same_family_of_event",
            vision={"feature_vector": {"scene_stable": 0.8, "utensil_on_floor": 1.0}, "hypothesis": "utensil_confirmed", "candidate_category": "cookware_fall", "context_labels": ["kitchen", "floor"]},
            audio={"feature_vector": {"ambient_hum": 0.1, "intensity": 0.1}, "hypothesis": "quiet_room", "context_labels": ["kitchen"]},
            time_context={"feature_vector": {"hour_night": 0.4, "context_shift": 0.1}, "hypothesis": "cleanup_resolution", "context_labels": ["time:evening", "task:cleaning", "kitchen"]},
            emotion=deepcopy(resolved_emotion),
        ),
    ]

    return {
        "surprise_then_resolution": surprise_then_resolution,
        "same_cue_later": same_cue_later,
        "false_cue_mismatch": false_cue_mismatch,
        "intrusive_repeated_poor_fit": intrusive_repeated_poor_fit,
        "repeated_pattern_abstraction": repeated_pattern_abstraction,
    }
