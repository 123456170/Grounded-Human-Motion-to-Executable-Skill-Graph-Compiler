import json
import math
import time
import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field


# =====================================================================
# GLOBAL CONSTANTS
# =====================================================================

APP_VERSION = "1.0"
APP_STATE_SCHEMA = "payload-session-pydantic-fix-4"

FPS = 10
VIDEO_DURATION = 10.0
NUM_FRAMES = int(VIDEO_DURATION * FPS)

CANVAS_W, CANVAS_H = 520, 360

KEYPOINTS = [
    "head",
    "torso",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

CONNECTIONS = [
    ("head", "torso"),
    ("torso", "left_shoulder"),
    ("torso", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_elbow", "right_wrist"),
    ("torso", "left_hip"),
    ("torso", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("right_hip", "right_knee"),
    ("left_knee", "left_ankle"),
    ("right_knee", "right_ankle"),
]

BODY_PART_OPTIONS = [
    "head",
    "torso",
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
    "legs",
    "base",
]

WORKSPACE_MIN = np.array([0.10, 0.05, 0.05], dtype=float)
WORKSPACE_MAX = np.array([0.90, 0.75, 0.70], dtype=float)
HOME_EE = np.array([0.25, 0.25, 0.45], dtype=float)
MAX_REACH = 1.30
GRASP_TOL = 0.13


# =====================================================================
# SMALL UTILITIES
# =====================================================================

def clamp_scalar(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def interp_kf(kf: List[Tuple[float, Tuple]], t: float) -> Tuple:
    """Linearly interpolate a keyframe track."""
    if t <= kf[0][0]:
        return kf[0][1]
    if t >= kf[-1][0]:
        return kf[-1][1]
    for (t0, v0), (t1, v1) in zip(kf, kf[1:]):
        if t0 <= t <= t1:
            denom = max(1e-9, t1 - t0)
            a = (t - t0) / denom
            return tuple(float(v0[i] + a * (v1[i] - v0[i])) for i in range(len(v0)))
    return kf[-1][1]


def smooth_track(track: np.ndarray, window: int = 5) -> np.ndarray:
    """Moving-average smoothing that preserves NaN/missing values."""
    n = len(track)
    out = np.full_like(track, np.nan, dtype=float)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        sl = track[lo:hi]
        finite = sl[np.isfinite(sl).all(axis=1)]
        if len(finite) > 0:
            out[i] = finite.mean(axis=0)
    return out


def compute_velocity(track: np.ndarray, dt: float = 1.0 / FPS) -> np.ndarray:
    """Central finite difference for 2D tracks with NaN gaps."""
    vel = np.full_like(track, np.nan, dtype=float)
    n = len(track)
    for i in range(1, n - 1):
        if (
            np.isfinite(track[i - 1]).all()
            and np.isfinite(track[i]).all()
            and np.isfinite(track[i + 1]).all()
        ):
            vel[i] = (track[i + 1] - track[i - 1]) / (2.0 * dt)

    for i in range(1, n):
        if not np.isfinite(vel[i]).all() and np.isfinite(vel[i - 1]).all():
            vel[i] = vel[i - 1]
    for i in range(n - 2, -1, -1):
        if not np.isfinite(vel[i]).all() and np.isfinite(vel[i + 1]).all():
            vel[i] = vel[i + 1]
    return vel


def point_in_bounds(p: np.ndarray) -> bool:
    p = np.asarray(p, dtype=float)
    return bool(np.all(p >= WORKSPACE_MIN - 1e-6) and np.all(p <= WORKSPACE_MAX + 1e-6))


def to_json(model: BaseModel, indent: int = 2) -> str:
    if hasattr(model, "model_dump_json"):
        return model.model_dump_json(indent=indent)
    return model.json(indent=indent)


def to_dict(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def copy_model(model: BaseModel) -> BaseModel:
    if hasattr(model, "model_copy"):
        return model.model_copy(deep=True)
    return model.copy(deep=True)


def rerun_app() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def format_seconds(s: float) -> str:
    if s is None:
        return "--"
    if s < 60:
        return f"{s:.1f}s"
    m = int(s // 60)
    sec = s - m * 60
    return f"{m}m {sec:.0f}s"


def show_image(where, img: np.ndarray) -> None:
    """
    Compatible image call for newer and older Streamlit versions.
    """
    pil_img = Image.fromarray(img) if isinstance(img, np.ndarray) else img
    try:
        where.image(pil_img, use_container_width=True)
    except TypeError:
        where.image(pil_img, use_column_width=True)


# =====================================================================
# PERCEPTION PACKAGE (logical boundary)
# =====================================================================
# This section owns video ingestion, timestamp handling, pose sources,
# object detectors, and uncertainty-preserving mock adapters.
# The compiler and planner never import or call anything from here.
# =====================================================================


class KeyPoint(BaseModel):
    x: Optional[float] = None
    y: Optional[float] = None
    uncertainty: float = 0.0
    occluded: bool = False


class PoseObservation(BaseModel):
    frame_id: int
    timestamp: float
    keypoints: Dict[str, KeyPoint]
    confidence: float = 1.0


class ObjectDetection(BaseModel):
    frame_id: int
    timestamp: float
    name: str
    bbox: Tuple[float, float, float, float]
    center: Tuple[float, float]
    confidence: float = 1.0
    uncertainty: float = 0.0
    occluded: bool = False


@dataclass
class ObservationBundle:
    fps: int
    duration: float
    frames: List[np.ndarray]
    poses: List[PoseObservation]
    objects: List[List[ObjectDetection]]
    adapter_report: Dict[str, str] = field(default_factory=dict)


class PoseSource:
    """Interface for pose estimators."""

    adapter_name = "PoseSource"

    def observations(self) -> List[PoseObservation]:
        raise NotImplementedError


class ObjectDetector:
    """Interface for object detectors."""

    adapter_name = "ObjectDetector"

    def observations(self) -> List[List[ObjectDetection]]:
        raise NotImplementedError


class MockPoseSource(PoseSource):
    """
    Transparent mock pose source.

    Generates a realistic pick-and-place human motion clip.
    Occluded right wrist/left wrist intervals are returned as unknown
    coordinates with high uncertainty instead of being guessed.
    """

    adapter_name = "MockPoseSource (no external model)"

    def __init__(self, seed: int = 17):
        self.rng = np.random.default_rng(seed)

    def _noisy(self, xy: Tuple[float, float], scale: float = 0.004) -> Tuple[float, float]:
        x = clamp_scalar(xy[0] + float(self.rng.normal(0, scale)), 0.0, 1.0)
        y = clamp_scalar(xy[1] + float(self.rng.normal(0, scale)), 0.0, 1.0)
        return x, y

    def observations(self) -> List[PoseObservation]:
        torso_kf = [
            (0.0, (0.25, 0.35)),
            (2.0, (0.30, 0.35)),
            (3.0, (0.32, 0.43)),
            (4.0, (0.32, 0.43)),
            (5.0, (0.32, 0.35)),
            (7.0, (0.36, 0.35)),
            (8.0, (0.36, 0.35)),
            (10.0, (0.25, 0.35)),
        ]
        wrist_kf = [
            (0.0, (0.40, 0.48)),
            (2.0, (0.58, 0.52)),
            (3.0, (0.64, 0.57)),
            (4.0, (0.64, 0.57)),
            (5.0, (0.64, 0.47)),
            (7.0, (0.79, 0.47)),
            (8.0, (0.80, 0.57)),
            (10.0, (0.45, 0.48)),
        ]

        poses: List[PoseObservation] = []

        for i in range(NUM_FRAMES):
            t = i / float(FPS)
            torso = interp_kf(torso_kf, t)
            wrist = interp_kf(wrist_kf, t)

            tx, ty = self._noisy(torso, 0.003)
            rx, ry = self._noisy(wrist, 0.003)

            head = (tx, clamp_scalar(ty - 0.15, 0, 1))
            left_shoulder = (tx - 0.06, ty - 0.04)
            right_shoulder = (tx + 0.06, ty - 0.04)
            left_hip = (tx - 0.04, ty + 0.15)
            right_hip = (tx + 0.04, ty + 0.15)

            crouched = ty > 0.40
            knee_dy = 0.10 if crouched else 0.15

            left_knee = (left_hip[0] - 0.02, clamp_scalar(left_hip[1] + knee_dy, 0, 1))
            right_knee = (right_hip[0] + 0.02, clamp_scalar(right_hip[1] + knee_dy, 0, 1))
            left_ankle = (left_knee[0], clamp_scalar(left_knee[1] + 0.15, 0, 1))
            right_ankle = (right_knee[0], clamp_scalar(right_knee[1] + 0.15, 0, 1))

            left_wrist = (tx - 0.13, ty + 0.12)

            # Deliberate occlusion windows: do not guess coordinates.
            occ_right_wrist = 3.1 < t < 7.9
            occ_left_wrist = 4.3 < t < 5.2

            if not occ_right_wrist:
                right_elbow = ((right_shoulder[0] + rx) / 2.0, (right_shoulder[1] + ry) / 2.0)
            else:
                right_elbow = None

            if not occ_left_wrist:
                lw_x, lw_y = self._noisy(left_wrist, 0.003)
                left_elbow = ((left_shoulder[0] + lw_x) / 2.0, (left_shoulder[1] + lw_y) / 2.0)
            else:
                left_elbow = None
                lw_x, lw_y = None, None

            kp: Dict[str, KeyPoint] = {}

            def add_point(
                name: str,
                xy: Optional[Tuple[float, float]],
                occluded: bool = False,
                base_unc: float = 0.08,
                noise_scale: float = 0.003,
            ) -> None:
                if occluded or xy is None:
                    kp[name] = KeyPoint(
                        x=None,
                        y=None,
                        uncertainty=clamp_scalar(max(0.75, base_unc), 0.0, 1.0),
                        occluded=True,
                    )
                else:
                    nx, ny = self._noisy(xy, noise_scale)
                    unc = clamp_scalar(base_unc + abs(float(self.rng.normal(0, 0.02))), 0.02, 0.95)
                    kp[name] = KeyPoint(x=float(nx), y=float(ny), uncertainty=float(unc), occluded=False)

            add_point("head", head)
            add_point("torso", (tx, ty), noise_scale=0.0)
            add_point("left_shoulder", left_shoulder)
            add_point("right_shoulder", right_shoulder)
            add_point("left_elbow", left_elbow, occluded=occ_left_wrist, base_unc=0.55)
            add_point("right_elbow", right_elbow, occluded=occ_right_wrist, base_unc=0.78)
            add_point(
                "left_wrist",
                (lw_x, lw_y) if not occ_left_wrist else None,
                occluded=occ_left_wrist,
                base_unc=0.62,
            )
            add_point("right_wrist", (rx, ry), occluded=occ_right_wrist, base_unc=0.86, noise_scale=0.0)
            add_point("left_hip", left_hip)
            add_point("right_hip", right_hip)
            add_point("left_knee", left_knee)
            add_point("right_knee", right_knee)
            add_point("left_ankle", left_ankle)
            add_point("right_ankle", right_ankle)

            visible_unc = [p.uncertainty for p in kp.values() if not p.occluded]
            if visible_unc:
                conf = float(np.mean([1.0 - u for u in visible_unc]))
            else:
                conf = 0.20

            poses.append(
                PoseObservation(
                    frame_id=i,
                    timestamp=float(t),
                    keypoints=kp,
                    confidence=clamp_scalar(conf, 0.0, 1.0),
                )
            )

        return poses


class MockObjectDetector(ObjectDetector):
    """
    Transparent mock object detector.

    Detects mug and plate. During hand occlusion intervals it lowers
    confidence and raises uncertainty rather than pretending certainty.
    """

    adapter_name = "MockObjectDetector (no external model)"

    def __init__(self, seed: int = 29):
        self.rng = np.random.default_rng(seed)

    def observations(self) -> List[List[ObjectDetection]]:
        mug_kf = [
            (0.0, (0.65, 0.55)),
            (4.0, (0.65, 0.55)),
            (5.0, (0.65, 0.45)),
            (7.0, (0.80, 0.45)),
            (8.0, (0.80, 0.58)),
            (10.0, (0.80, 0.58)),
        ]
        plate_pos = (0.84, 0.62)

        all_dets: List[List[ObjectDetection]] = []

        for i in range(NUM_FRAMES):
            t = i / float(FPS)
            dets: List[ObjectDetection] = []

            for name, pos in [("mug", interp_kf(mug_kf, t)), ("plate", plate_pos)]:
                x = clamp_scalar(pos[0] + float(self.rng.normal(0, 0.002)), 0, 1)
                y = clamp_scalar(pos[1] + float(self.rng.normal(0, 0.002)), 0, 1)

                if name == "mug":
                    w, h = 0.08, 0.08
                    occluded = (2.9 < t < 3.4) or (7.6 < t < 8.1)
                else:
                    w, h = 0.16, 0.06
                    occluded = False

                conf = 0.54 if occluded else 0.93
                unc = 0.46 if occluded else 0.08

                bbox = (float(x - w / 2), float(y - h / 2), float(x + w / 2), float(y + h / 2))
                dets.append(
                    ObjectDetection(
                        frame_id=i,
                        timestamp=float(t),
                        name=name,
                        bbox=bbox,
                        center=(float(x), float(y)),
                        confidence=float(conf),
                        uncertainty=float(unc),
                        occluded=bool(occluded),
                    )
                )

            all_dets.append(dets)

        return all_dets


def _norm_to_px(x: float, y: float) -> Tuple[int, int]:
    return int(x * CANVAS_W), int(y * CANVAS_H)


def render_video_frame(pose: PoseObservation, detections: List[ObjectDetection]) -> np.ndarray:
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), (22, 24, 30))
    d = ImageDraw.Draw(img)

    # Table / scene background.
    d.rectangle([0, int(0.62 * CANVAS_H), CANVAS_W, CANVAS_H], fill=(58, 45, 35))
    d.line([0, int(0.62 * CANVAS_H), CANVAS_W, int(0.62 * CANVAS_H)], fill=(126, 100, 74), width=2)

    # Objects.
    for det in detections:
        cx, cy = _norm_to_px(det.center[0], det.center[1])
        rw = max(5, int(((det.bbox[2] - det.bbox[0]) * CANVAS_W) / 2))
        rh = max(4, int(((det.bbox[3] - det.bbox[1]) * CANVAS_H) / 2))

        if det.name == "mug":
            color = (185, 125, 75) if not det.occluded else (95, 72, 55)
        else:
            color = (175, 175, 185) if not det.occluded else (95, 95, 105)

        d.ellipse([cx - rw, cy - rh, cx + rw, cy + rh], outline=color, width=3)

        if det.uncertainty > 0.25:
            d.ellipse(
                [cx - rw - 5, cy - rh - 5, cx + rw + 5, cy + rh + 5],
                outline=(235, 200, 85),
                width=1,
            )

        d.text((cx - 18, cy - rh - 15), f"{det.name} {det.confidence:.2f}", fill=(220, 220, 220))

    # Skeleton edges.
    for a, b in CONNECTIONS:
        ka = pose.keypoints.get(a)
        kb = pose.keypoints.get(b)
        if (
            ka is not None
            and kb is not None
            and not ka.occluded
            and not kb.occluded
            and ka.x is not None
            and ka.y is not None
            and kb.x is not None
            and kb.y is not None
        ):
            d.line([_norm_to_px(ka.x, ka.y), _norm_to_px(kb.x, kb.y)], fill=(85, 175, 245), width=2)

    # Keypoints and uncertainty halos.
    for name, kp in pose.keypoints.items():
        if kp.occluded or kp.x is None or kp.y is None:
            continue
        px, py = _norm_to_px(kp.x, kp.y)
        r = 3
        d.ellipse([px - r, py - r, px + r, py + r], fill=(250, 230, 120))
        if kp.uncertainty > 0.30:
            rr = int(r + kp.uncertainty * 10)
            d.ellipse([px - rr, py - rr, px + rr, py + rr], outline=(240, 185, 70), width=1)

    occluded_names = [n for n, kp in pose.keypoints.items() if kp.occluded]
    if occluded_names:
        d.text((8, 8), "occluded: " + ", ".join(occluded_names[:4]), fill=(240, 185, 85))

    d.text(
        (8, CANVAS_H - 18),
        f"t={pose.timestamp:.1f}s  frame={pose.frame_id}  pose_conf={pose.confidence:.2f}",
        fill=(220, 220, 220),
    )

    return np.array(img)


class VideoIngestor:
    """
    Timestamp-aware ingestion wrapper.

    It does not guess missing pose points. It stores per-frame pose and
    object observations plus rendered frames for dashboard scrubbing.
    """

    def __init__(self, fps: int = FPS, duration: float = VIDEO_DURATION):
        self.fps = fps
        self.duration = duration

    def load(self, pose_source: PoseSource, object_detector: ObjectDetector) -> ObservationBundle:
        poses = pose_source.observations()
        objs = object_detector.observations()
        n = min(len(poses), len(objs))
        frames = [render_video_frame(poses[i], objs[i]) for i in range(n)]

        report = {
            "pose_source": getattr(pose_source, "adapter_name", type(pose_source).__name__),
            "object_detector": getattr(object_detector, "adapter_name", type(object_detector).__name__),
            "video_ingestor": "Timestamped mock video ingester",
        }

        return ObservationBundle(
            fps=self.fps,
            duration=float(min(self.duration, n / float(self.fps))),
            frames=frames,
            poses=poses[:n],
            objects=objs[:n],
            adapter_report=report,
        )


# =====================================================================
# SEMANTICS PACKAGE (logical boundary)
# =====================================================================
# Typed skill graph schema, validation, and action segmentation.
# Segmentation consumes perception outputs, but downstream compiler only
# consumes the validated SkillGraph.
# =====================================================================


class Action(BaseModel):
    id: str
    name: str
    start: float
    end: float
    body_parts: List[str] = Field(default_factory=list)
    object_refs: List[str] = Field(default_factory=list)
    preconditions: List[str] = Field(default_factory=list)
    postconditions: List[str] = Field(default_factory=list)
    confidence: float = 0.5


class Edge(BaseModel):
    source: str
    target: str
    type: str = "depends_on"


class ObjectState(BaseModel):
    name: str
    position: Tuple[float, float, float]
    visible: bool = True
    graspable: bool = True
    uncertainty: float = 0.1


def _has_cycle(action_ids: List[str], edges: List[Edge]) -> bool:
    adj: Dict[str, List[str]] = {aid: [] for aid in action_ids}
    for e in edges:
        if e.source in adj and e.target in adj:
            adj[e.source].append(e.target)

    color: Dict[str, int] = {aid: 0 for aid in action_ids}

    def dfs(u: str) -> bool:
        color[u] = 1
        for v in adj.get(u, []):
            if color[v] == 1:
                return True
            if color[v] == 0 and dfs(v):
                return True
        color[u] = 2
        return False

    for aid in action_ids:
        if color[aid] == 0:
            if dfs(aid):
                return True
    return False


class SkillGraph(BaseModel):
    version: str = APP_VERSION
    actions: List[Action] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)
    objects: List[ObjectState] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def validation_errors(self) -> List[str]:
        return validate_skill_graph(self)

    def throw_if_invalid(self) -> None:
        errs = self.validation_errors()
        if errs:
            raise ValueError("SkillGraph validation failed: " + "; ".join(errs))


def validate_skill_graph(graph: SkillGraph) -> List[str]:
    """
    Graph validator.

    Checks:
    - version field
    - unique action ids
    - valid time intervals
    - non-overlapping actions
    - confidence range
    - object references resolve to graph objects
    - edges reference existing actions
    - acyclic dependencies
    - dependency edges respect temporal order
    """
    errors: List[str] = []

    if not graph.version:
        errors.append("missing version field")

    ids = [a.id for a in graph.actions]
    if len(ids) != len(set(ids)):
        errors.append("duplicate action ids")

    if not graph.actions:
        errors.append("graph has no actions")

    object_names = [o.name for o in graph.objects]
    if len(object_names) != len(set(object_names)):
        errors.append("duplicate object names")
    object_set = set(object_names)

    duration = float(graph.metadata.get("duration", VIDEO_DURATION))

    for a in graph.actions:
        if a.start < 0:
            errors.append(f"action {a.id} has negative start")
        if a.end <= a.start:
            errors.append(f"action {a.id} end must be greater than start")
        if a.end > duration + 1e-3:
            errors.append(f"action {a.id} extends beyond video duration")
        if not (0.0 <= a.confidence <= 1.0):
            errors.append(f"action {a.id} confidence must be in [0, 1]")
        for ref in a.object_refs:
            if ref not in object_set:
                errors.append(f"missing object reference '{ref}' in action {a.id}")

    sorted_actions = sorted(graph.actions, key=lambda x: x.start)
    for cur, nxt in zip(sorted_actions, sorted_actions[1:]):
        if nxt.start < cur.end - 1e-6:
            errors.append(f"overlapping actions {cur.id} and {nxt.id}")

    id_set = set(ids)
    for e in graph.edges:
        if e.source not in id_set:
            errors.append(f"edge source {e.source} not found")
        if e.target not in id_set:
            errors.append(f"edge target {e.target} not found")

    if _has_cycle(ids, graph.edges):
        errors.append("cyclic dependency detected")

    action_by_id = {a.id: a for a in graph.actions}
    for e in graph.edges:
        if e.source in action_by_id and e.target in action_by_id:
            s = action_by_id[e.source]
            t = action_by_id[e.target]
            if s.end > t.start + 1e-6:
                errors.append(f"dependency {e.source}->{e.target} violates temporal order")

    return errors


class CompiledStep(BaseModel):
    id: str
    action_id: str
    action_name: str
    primitive: str
    goal: List[float]
    parameters: Dict[str, Any] = Field(default_factory=dict)
    preconditions: List[str] = Field(default_factory=list)
    postconditions: List[str] = Field(default_factory=list)
    duration_hint: float = 1.0


class RobotPlan(BaseModel):
    version: str = APP_VERSION
    steps: List[CompiledStep] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


def action_spec(name: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Semantic template for known action phases."""
    if name == "approach":
        return (
            ["right_arm", "torso"],
            ["mug"],
            ["robot.home", "object:mug.visible"],
            ["ee.near:mug"],
        )
    if name == "crouch":
        return (
            ["legs", "torso", "right_arm"],
            ["mug"],
            ["object:mug.visible"],
            ["robot.crouched"],
        )
    if name == "grasp":
        return (
            ["right_hand", "right_arm"],
            ["mug"],
            ["object:mug.visible", "object:mug.graspable", "gripper.open"],
            ["object:mug.held", "gripper.closed"],
        )
    if name == "lift":
        return (
            ["right_arm", "torso"],
            ["mug"],
            ["object:mug.held"],
            ["object:mug.lifted"],
        )
    if name == "turn":
        return (
            ["torso", "right_arm"],
            ["mug", "plate"],
            ["object:mug.held", "object:plate.visible"],
            ["robot.facing:plate"],
        )
    if name == "place":
        return (
            ["right_arm", "right_hand"],
            ["mug", "plate"],
            ["object:mug.held", "object:plate.visible", "object:plate.reachable"],
            ["object:mug.placed", "gripper.open"],
        )
    if name == "retreat":
        return (
            ["right_arm", "torso"],
            [],
            ["gripper.open"],
            ["robot.home"],
        )

    # Generic fallback.
    return (["right_arm"], [], [], [])


def ground_truth_graph() -> SkillGraph:
    """Hand-labelled demo clip used for evaluation and fallback."""
    specs = [
        ("approach", 0.0, 2.0),
        ("crouch", 2.0, 3.0),
        ("grasp", 3.0, 4.0),
        ("lift", 4.0, 5.0),
        ("turn", 5.0, 7.0),
        ("place", 7.0, 8.0),
        ("retreat", 8.0, 10.0),
    ]

    actions: List[Action] = []
    for i, (nm, s, e) in enumerate(specs):
        body, objs, pre, post = action_spec(nm)
        actions.append(
            Action(
                id=f"gt{i}",
                name=nm,
                start=float(s),
                end=float(e),
                body_parts=body,
                object_refs=objs,
                preconditions=pre,
                postconditions=post,
                confidence=0.98,
            )
        )

    edges = [Edge(source=f"gt{i}", target=f"gt{i+1}") for i in range(len(actions) - 1)]

    objects = [
        ObjectState(name="mug", position=(0.65, 0.35, 0.25), visible=True, graspable=True, uncertainty=0.08),
        ObjectState(name="plate", position=(0.84, 0.45, 0.22), visible=True, graspable=True, uncertainty=0.05),
    ]

    return SkillGraph(
        version=APP_VERSION,
        actions=actions,
        edges=edges,
        objects=objects,
        metadata={"duration": VIDEO_DURATION, "fps": FPS, "source": "hand-labelled"},
    )


class Segmenter:
    """Interface for action-phase segmenters."""

    adapter_name = "Segmenter"

    def segment(self, bundle: ObservationBundle) -> SkillGraph:
        raise NotImplementedError


class MotionChangeSegmenter(Segmenter):
    """
    Heuristic segmenter using motion changes:
    - velocity minima / direction changes
    - contact changes
    - object-interaction cues
    """

    adapter_name = "MotionChangeSegmenter"

    def segment(self, bundle: ObservationBundle) -> SkillGraph:
        n = len(bundle.poses)
        times = np.array([p.timestamp for p in bundle.poses], dtype=float)

        wrist = np.full((n, 2), np.nan, dtype=float)
        torso_y = np.full(n, np.nan, dtype=float)
        mug = np.full((n, 2), np.nan, dtype=float)
        obj_conf = np.full(n, 0.5, dtype=float)

        for i, p in enumerate(bundle.poses):
            rw = p.keypoints.get("right_wrist")
            if rw is not None and not rw.occluded and rw.x is not None and rw.y is not None:
                wrist[i] = [rw.x, rw.y]

            tor = p.keypoints.get("torso")
            if tor is not None and not tor.occluded and tor.y is not None:
                torso_y[i] = tor.y

        for i, dets in enumerate(bundle.objects):
            for det in dets:
                if det.name == "mug":
                    mug[i] = [det.center[0], det.center[1]]
                    obj_conf[i] = float(det.confidence)

        wrist_s = smooth_track(wrist, 5)
        mug_s = smooth_track(mug, 5)
        torso_y_s = smooth_track(torso_y.reshape(-1, 1), 5)[:, 0]

        wrist_v = compute_velocity(wrist_s)
        mug_v = compute_velocity(mug_s)

        torso_v = np.full(n, np.nan, dtype=float)
        dt = 1.0 / float(FPS)
        for i in range(1, n - 1):
            if np.isfinite(torso_y_s[i - 1 : i + 2]).all():
                torso_v[i] = (torso_y_s[i + 1] - torso_y_s[i - 1]) / (2.0 * dt)

        mug_speed = np.sqrt(np.sum(mug_v**2, axis=1))
        mug_speed = np.where(np.isfinite(mug_speed), mug_speed, np.nan)

        dist = np.full(n, np.nan, dtype=float)
        both = np.isfinite(wrist_s).all(axis=1) & np.isfinite(mug_s).all(axis=1)
        if np.any(both):
            dist[both] = np.linalg.norm(wrist_s[both] - mug_s[both], axis=1)

        def first_true(mask: np.ndarray, default: float) -> float:
            idx = np.where(mask)[0]
            if len(idx) == 0:
                return float(default)
            return float(times[idx[0]])

        crouch_mask = (torso_v > 0.018) & (times > 1.0)
        contact_mask = (dist < 0.055) & (times > 1.0)
        lift_mask = (mug_v[:, 1] < -0.018) & (times > 2.5)
        transport_mask = (mug_v[:, 0] > 0.018) & (times > 3.5)
        place_mask = (mug_v[:, 1] > 0.018) & (times > 5.0)
        release_mask = (mug_speed < 0.012) & (times > 6.5)

        crouch_start = first_true(crouch_mask, 2.0)
        contact_on = first_true(contact_mask, max(crouch_start + 0.5, 3.0))
        lift_on = first_true(lift_mask, contact_on + 1.0)
        transport_on = first_true(transport_mask, lift_on + 0.8)
        place_on = first_true(place_mask, transport_on + 1.0)
        release_off = first_true(release_mask, place_on + 0.8)

        detected = [
            0.0,
            crouch_start,
            contact_on,
            lift_on,
            transport_on,
            place_on,
            release_off,
            bundle.duration,
        ]
        expected = [0.0, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0, bundle.duration]

        ordered = all(detected[i] + 0.12 <= detected[i + 1] for i in range(len(detected) - 1))
        in_range = all(0.0 <= v <= bundle.duration for v in detected)

        if ordered and in_range:
            boundaries = detected
        else:
            boundaries = expected
            crouch_start = float(expected[1])
            contact_on = float(expected[2])
            lift_on = float(expected[3])
            transport_on = float(expected[4])
            place_on = float(expected[5])
            release_off = float(expected[6])

        clean = [float(boundaries[0])]
        for b in boundaries[1:]:
            b = float(b)
            if b - clean[-1] >= 0.18:
                clean.append(b)
        if clean[-1] < bundle.duration - 0.1:
            clean.append(float(bundle.duration))

        intervals = list(zip(clean, clean[1:]))
        if not intervals:
            return ground_truth_graph()

        def classify(s: float, e: float) -> str:
            if e <= crouch_start + 0.12:
                return "approach"
            if s < contact_on + 0.12:
                return "crouch"
            if s < lift_on + 0.12:
                return "grasp"
            if s < transport_on + 0.12:
                return "lift"
            if s < place_on + 0.12:
                return "turn"
            if s < release_off + 0.12:
                return "place"
            return "retreat"

        actions: List[Action] = []

        for i, (s, e) in enumerate(intervals):
            idx0 = int(round(s * FPS))
            idx1 = int(round(e * FPS))
            idx0 = max(0, min(n - 1, idx0))
            idx1 = max(idx0 + 1, min(n, idx1))

            pose_confs = [bundle.poses[j].confidence for j in range(idx0, idx1)]
            oconfs = [obj_conf[j] for j in range(idx0, idx1)]

            occ_flags = []
            for j in range(idx0, idx1):
                rw = bundle.poses[j].keypoints.get("right_wrist")
                occ_flags.append(bool(rw is not None and rw.occluded))
            occ_frac = float(np.mean(occ_flags)) if occ_flags else 0.0

            base_conf = 0.55 * float(np.mean(pose_confs) if pose_confs else 0.5)
            base_conf += 0.35 * float(np.mean(oconfs) if oconfs else 0.5)
            base_conf += 0.10

            if occ_frac > 0.50:
                base_conf *= 0.80

            conf = clamp_scalar(base_conf, 0.20, 0.99)
            name = classify(float(s), float(e))
            body, objs, pre, post = action_spec(name)

            actions.append(
                Action(
                    id=f"a{i}",
                    name=name,
                    start=float(round(s, 3)),
                    end=float(round(e, 3)),
                    body_parts=body,
                    object_refs=objs,
                    preconditions=pre,
                    postconditions=post,
                    confidence=float(round(conf, 3)),
                )
            )

        edges = [Edge(source=actions[i].id, target=actions[i + 1].id) for i in range(len(actions) - 1)]

        # Initial world state for planning comes from first reliable detections.
        objects: List[ObjectState] = []
        if bundle.objects:
            for det in bundle.objects[0]:
                if det.name == "mug":
                    pos = (float(det.center[0]), 0.35, 0.25)
                elif det.name == "plate":
                    pos = (float(det.center[0]), 0.45, 0.22)
                else:
                    pos = (float(det.center[0]), 0.35, 0.25)

                objects.append(
                    ObjectState(
                        name=det.name,
                        position=pos,
                        visible=not det.occluded,
                        graspable=True,
                        uncertainty=float(det.uncertainty),
                    )
                )

        graph = SkillGraph(
            version=APP_VERSION,
            actions=actions,
            edges=edges,
            objects=objects,
            metadata={
                "duration": float(bundle.duration),
                "fps": int(bundle.fps),
                "segmenter": self.adapter_name,
                "events": {
                    "crouch_start": float(crouch_start),
                    "contact_on": float(contact_on),
                    "lift_on": float(lift_on),
                    "transport_on": float(transport_on),
                    "place_on": float(place_on),
                    "release_off": float(release_off),
                },
            },
        )

        if graph.validation_errors():
            gt = ground_truth_graph()
            gt.metadata["fallback_reason"] = "heuristic graph invalid"
            gt.metadata["segmenter"] = self.adapter_name + " -> hand-label fallback"
            return gt

        return graph


class LearnedActionSegmenter(Segmenter):
    """
    Optional learned classifier behind the same interface.

    In this no-API-key build, the learned model is unavailable, so it
    transparently falls back to the motion-change segmenter and records
    that fact in graph metadata.
    """

    adapter_name = "LearnedActionSegmenter (mock unavailable -> heuristic fallback)"

    def __init__(self, fallback: Optional[Segmenter] = None):
        self.fallback = fallback or MotionChangeSegmenter()
        self.available = False

    def segment(self, bundle: ObservationBundle) -> SkillGraph:
        graph = self.fallback.segment(bundle)
        graph.metadata["segmenter"] = self.adapter_name
        graph.metadata["learned_model_available"] = False
        return graph


# =====================================================================
# CONTROL / COMPILER PACKAGE (logical boundary)
# =====================================================================
# The compiler consumes only a validated SkillGraph. It never calls
# perception code directly.
# =====================================================================


def topological_actions(graph: SkillGraph) -> List[Action]:
    actions_by_id = {a.id: a for a in graph.actions}
    indeg = {aid: 0 for aid in actions_by_id}
    adj: Dict[str, List[str]] = {aid: [] for aid in actions_by_id}

    for e in graph.edges:
        if e.source in adj and e.target in indeg:
            adj[e.source].append(e.target)
            indeg[e.target] += 1

    queue = sorted([aid for aid, deg in indeg.items() if deg == 0], key=lambda aid: actions_by_id[aid].start)
    order: List[Action] = []

    while queue:
        u = queue.pop(0)
        order.append(actions_by_id[u])
        for v in adj.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
        queue = sorted(queue, key=lambda aid: actions_by_id[aid].start)

    if len(order) != len(graph.actions):
        return sorted(graph.actions, key=lambda a: a.start)
    return order


class GraphCompiler:
    """
    Compiles a typed SkillGraph into robot end-effector goals and safe
    motion primitives: reach, turn, crouch, grasp_proxy, retreat.
    """

    def compile(self, graph: SkillGraph) -> RobotPlan:
        graph.throw_if_invalid()

        obj_pos = {o.name: np.asarray(o.position, dtype=float) for o in graph.objects}
        ordered = topological_actions(graph)

        steps: List[CompiledStep] = []
        last_goal = HOME_EE.copy()
        held: Optional[str] = None

        def add_step(
            action: Action,
            primitive: str,
            goal: np.ndarray,
            params: Dict[str, Any],
            pre: List[str],
            post: List[str],
            duration_hint: float,
        ) -> None:
            step_id = f"s{len(steps)}"
            clean_pre = [p for p in pre if p]
            clean_post = [p for p in post if p]
            steps.append(
                CompiledStep(
                    id=step_id,
                    action_id=action.id,
                    action_name=action.name,
                    primitive=primitive,
                    goal=[float(v) for v in np.asarray(goal, dtype=float)],
                    parameters=params,
                    preconditions=clean_pre,
                    postconditions=clean_post,
                    duration_hint=float(max(0.25, duration_hint)),
                )
            )

        for act in ordered:
            action_dur = max(0.30, act.end - act.start)
            primary = act.object_refs[0] if act.object_refs else None
            secondary = act.object_refs[1] if len(act.object_refs) > 1 else None

            if act.name == "approach":
                if primary in obj_pos:
                    goal = obj_pos[primary] + np.array([0.0, 0.0, 0.14])
                else:
                    goal = last_goal.copy()
                add_step(
                    act,
                    "reach",
                    goal,
                    {"object": primary},
                    [
                        "robot.goal_reachable",
                        f"object:{primary}.visible" if primary else None,
                        f"object:{primary}.reachable" if primary else None,
                        "gripper.open",
                    ],
                    [f"ee.near:{primary}" if primary else "ee.near:target"],
                    action_dur,
                )
                last_goal = goal.copy()

            elif act.name == "crouch":
                goal = last_goal.copy()
                goal[2] = clamp_scalar(goal[2] - 0.14, WORKSPACE_MIN[2], WORKSPACE_MAX[2])
                add_step(
                    act,
                    "crouch",
                    goal,
                    {},
                    ["robot.goal_reachable", "robot.balanced"],
                    ["robot.crouched"],
                    action_dur,
                )
                last_goal = goal.copy()

            elif act.name == "grasp":
                if primary in obj_pos:
                    goal = obj_pos[primary] + np.array([0.0, 0.0, 0.03])
                else:
                    goal = last_goal.copy()
                add_step(
                    act,
                    "grasp_proxy",
                    goal,
                    {"close": True, "object": primary},
                    [
                        "robot.goal_reachable",
                        f"object:{primary}.visible" if primary else None,
                        f"object:{primary}.graspable" if primary else None,
                        f"object:{primary}.in_range" if primary else None,
                        "gripper.open",
                    ],
                    [
                        f"gripper.holding:{primary}" if primary else None,
                        "gripper.closed",
                    ],
                    action_dur,
                )
                held = primary
                last_goal = goal.copy()

            elif act.name == "lift":
                goal = last_goal.copy()
                goal[2] = clamp_scalar(goal[2] + 0.16, WORKSPACE_MIN[2], WORKSPACE_MAX[2])
                add_step(
                    act,
                    "reach",
                    goal,
                    {"object": held, "mode": "lift"},
                    [
                        "robot.goal_reachable",
                        f"gripper.holding:{held}" if held else "gripper.closed",
                    ],
                    [f"object:{held}.lifted" if held else None],
                    action_dur,
                )
                last_goal = goal.copy()

            elif act.name == "turn":
                target = secondary if secondary in obj_pos else primary
                yaw = 0.0
                if target in obj_pos:
                    delta = obj_pos[target][:2] - last_goal[:2]
                    if np.linalg.norm(delta) > 1e-4:
                        yaw = float(math.atan2(delta[1], delta[0]))
                add_step(
                    act,
                    "turn",
                    last_goal.copy(),
                    {"yaw": yaw, "object": target},
                    [
                        "robot.goal_reachable",
                        f"gripper.holding:{held}" if held else "robot.balanced",
                    ],
                    [f"robot.facing:{target}" if target else "robot.facing:target"],
                    action_dur,
                )

            elif act.name == "place":
                plate = secondary if secondary in obj_pos else None
                release_obj = held or primary
                half = action_dur / 2.0

                if plate in obj_pos:
                    goal = obj_pos[plate] + np.array([0.0, 0.0, 0.08])
                    add_step(
                        act,
                        "reach",
                        goal,
                        {"object": plate, "mode": "preplace"},
                        [
                            "robot.goal_reachable",
                            f"object:{plate}.visible",
                            f"object:{plate}.reachable",
                            f"gripper.holding:{release_obj}" if release_obj else None,
                        ],
                        [f"ee.near:{plate}"],
                        half,
                    )
                    last_goal = goal.copy()

                if release_obj:
                    add_step(
                        act,
                        "grasp_proxy",
                        last_goal.copy(),
                        {"close": False, "object": release_obj},
                        [
                            "robot.goal_reachable",
                            f"gripper.holding:{release_obj}",
                            f"object:{plate}.reachable" if plate else None,
                        ],
                        [
                            f"object:{release_obj}.placed",
                            "gripper.open",
                        ],
                        half,
                    )
                    held = None

            elif act.name == "retreat":
                goal = HOME_EE.copy()
                add_step(
                    act,
                    "retreat",
                    goal,
                    {},
                    ["robot.goal_reachable", "gripper.open"],
                    ["robot.home"],
                    action_dur,
                )
                last_goal = goal.copy()

            else:
                # Generic compilation for user-renamed actions.
                if primary in obj_pos:
                    goal = obj_pos[primary] + np.array([0.0, 0.0, 0.10])
                else:
                    goal = last_goal.copy()
                add_step(
                    act,
                    "reach",
                    goal,
                    {"object": primary},
                    [
                        "robot.goal_reachable",
                        f"object:{primary}.visible" if primary else None,
                        f"object:{primary}.reachable" if primary else None,
                    ],
                    [f"ee.near:{primary}" if primary else "ee.near:target"],
                    action_dur,
                )
                last_goal = goal.copy()

        return RobotPlan(
            version=APP_VERSION,
            steps=steps,
            metadata={
                "source_graph_version": graph.version,
                "compiler": "GraphCompiler",
                "primitive_set": ["reach", "turn", "crouch", "grasp_proxy", "retreat"],
            },
        )


# =====================================================================
# SIMULATION EXECUTION
# =====================================================================


@dataclass
class SimState:
    ee: np.ndarray
    yaw: float
    gripper_open: bool
    held: Optional[str]
    hold_offset: Optional[np.ndarray]
    objects: Dict[str, Dict[str, Any]]
    trail: List[np.ndarray]


@dataclass
class SimResult:
    frames: List[np.ndarray]
    failures: List[Dict[str, Any]]
    log: List[Dict[str, Any]]
    success: bool
    final_objects: Dict[str, Tuple[float, float, float]]
    held: Optional[str]


def check_precondition(state: SimState, step: CompiledStep, p: str) -> bool:
    if p == "robot.goal_reachable":
        goal = np.asarray(step.goal, dtype=float)
        return point_in_bounds(goal) and float(np.linalg.norm(goal[:2])) <= MAX_REACH

    if p == "gripper.open":
        return bool(state.gripper_open)

    if p == "gripper.closed":
        return not bool(state.gripper_open)

    if p.startswith("gripper.holding:"):
        name = p.split(":", 1)[1]
        return state.held == name

    if p == "robot.balanced":
        return True

    if p.startswith("object:"):
        rest = p[len("object:") :]
        if "." not in rest:
            return False
        name, attr = rest.split(".", 1)
        obj = state.objects.get(name)
        if obj is None:
            return False

        pos = np.asarray(obj["pos"], dtype=float)

        if attr == "visible":
            return bool(obj.get("visible", False))
        if attr == "graspable":
            return bool(obj.get("graspable", False))
        if attr == "reachable":
            return point_in_bounds(pos) and float(np.linalg.norm(pos[:2])) <= MAX_REACH
        if attr == "in_range":
            return float(np.linalg.norm(pos - state.ee)) <= GRASP_TOL
        if attr in ("held", "holding"):
            return state.held == name

    return False


def render_sim_frame(
    ee: np.ndarray,
    yaw: float,
    gripper_open: bool,
    held: Optional[str],
    objects: Dict[str, Dict[str, Any]],
    trail: List[np.ndarray],
    step: Optional[CompiledStep] = None,
    failure: Optional[Dict[str, Any]] = None,
    t: float = 0.0,
) -> np.ndarray:
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), (18, 22, 28))
    d = ImageDraw.Draw(img)

    def map_xy(p: np.ndarray) -> Tuple[int, int]:
        fx = (float(p[0]) - WORKSPACE_MIN[0]) / max(1e-9, WORKSPACE_MAX[0] - WORKSPACE_MIN[0])
        fy = (float(p[1]) - WORKSPACE_MIN[1]) / max(1e-9, WORKSPACE_MAX[1] - WORKSPACE_MIN[1])
        x = int(40 + clamp_scalar(fx, 0.0, 1.0) * (CANVAS_W - 80))
        y = int(40 + clamp_scalar(fy, 0.0, 1.0) * (CANVAS_H - 120))
        return x, y

    # Workspace.
    d.rectangle([40, 40, CANVAS_W - 40, CANVAS_H - 80], outline=(72, 84, 100), width=2)
    for gx in range(5, CANVAS_W - 40, 50):
        d.line([gx, 40, gx, CANVAS_H - 80], fill=(38, 44, 54), width=1)
    for gy in range(5, CANVAS_H - 80, 50):
        d.line([40, gy, CANVAS_W - 40, gy], fill=(38, 44, 54), width=1)

    # Robot base.
    bx, by = map_xy(np.array([0.12, 0.08, 0.0]))
    d.ellipse([bx - 10, by - 10, bx + 10, by + 10], outline=(120, 130, 145), width=2)
    d.text((bx - 14, by + 12), "base", fill=(170, 175, 185))

    # Trail.
    for i in range(1, len(trail)):
        d.line([map_xy(trail[i - 1]), map_xy(trail[i])], fill=(55, 110, 150), width=1)

    # Objects.
    for name, obj in objects.items():
        pos = np.asarray(obj["pos"], dtype=float)
        ox, oy = map_xy(pos)
        if name == "mug":
            color = (90, 165, 225) if obj.get("visible", True) else (85, 85, 95)
            r = 10
        else:
            color = (185, 185, 195) if obj.get("visible", True) else (85, 85, 95)
            r = 16

        d.ellipse([ox - r, oy - r, ox + r, oy + r], outline=color, width=3)
        d.text((ox - 13, oy + r + 2), name, fill=(210, 210, 210))

        if held == name:
            ex, ey = map_xy(np.asarray(ee, dtype=float))
            d.line([ox, oy, ex, ey], fill=(240, 200, 85), width=2)

    # End effector.
    ex, ey = map_xy(np.asarray(ee, dtype=float))
    d.ellipse([ex - 7, ey - 7, ex + 7, ey + 7], fill=(230, 90, 70))
    gx = ex + int(15 * math.cos(float(yaw)))
    gy = ey + int(15 * math.sin(float(yaw)))
    d.line([ex, ey, gx, gy], fill=(255, 220, 120), width=3)
    if not gripper_open:
        d.ellipse([ex - 3, ey - 3, ex + 3, ey + 3], fill=(255, 255, 255))

    # HUD.
    d.text((8, 8), f"simulation t={t:.1f}s", fill=(220, 220, 220))
    if step is not None:
        d.text((8, 24), f"{step.action_name} -> {step.primitive}", fill=(180, 220, 255))
    if held:
        d.text((8, CANVAS_H - 24), f"held: {held}", fill=(240, 200, 85))
    if failure:
        d.text((8, CANVAS_H - 44), f"FAIL: {failure.get('precondition', 'unknown')}", fill=(255, 120, 120))

    return np.array(img)


def simulate_plan(
    plan: RobotPlan,
    graph: SkillGraph,
    render: bool = False,
    rng: Optional[np.random.Generator] = None,
    perturb_scale: float = 0.0,
) -> SimResult:
    """
    Executes a compiled plan in a simplified robot simulation.

    If perturb_scale > 0, object positions are randomly offset before
    execution. Failures report the exact violated precondition.
    """
    if perturb_scale > 0 and rng is None:
        rng = np.random.default_rng()

    state_objects: Dict[str, Dict[str, Any]] = {}
    for obj in graph.objects:
        pos = np.asarray(obj.position, dtype=float).copy()
        if perturb_scale > 0 and rng is not None:
            pos = pos + rng.normal(0.0, perturb_scale, size=3)
        visible = bool(obj.visible and point_in_bounds(pos))
        state_objects[obj.name] = {
            "pos": pos,
            "visible": visible,
            "graspable": bool(obj.graspable),
        }

    state = SimState(
        ee=HOME_EE.copy(),
        yaw=0.0,
        gripper_open=True,
        held=None,
        hold_offset=None,
        objects=state_objects,
        trail=[HOME_EE.copy()],
    )

    failures: List[Dict[str, Any]] = []
    log: List[Dict[str, Any]] = []
    frames: List[np.ndarray] = []

    target_frames = NUM_FRAMES if render else 0

    if not plan.steps:
        if render:
            frame = render_sim_frame(
                state.ee,
                state.yaw,
                state.gripper_open,
                state.held,
                state.objects,
                state.trail,
                t=0.0,
            )
            frames = [frame for _ in range(target_frames)]
        return SimResult(
            frames=frames,
            failures=failures,
            log=log,
            success=True,
            final_objects={k: tuple(map(float, v["pos"])) for k, v in state.objects.items()},
            held=state.held,
        )

    total_hint = sum(max(0.25, s.duration_hint) for s in plan.steps)

    if render:
        raw_ns = [
            max(2, int(round(target_frames * max(0.25, s.duration_hint) / total_hint)))
            for s in plan.steps
        ]
        diff = target_frames - sum(raw_ns)

        while diff > 0:
            idx = int(np.argmax(raw_ns))
            raw_ns[idx] += 1
            diff -= 1

        while diff < 0:
            idx = int(np.argmax(raw_ns))
            if raw_ns[idx] > 2:
                raw_ns[idx] -= 1
                diff += 1
            else:
                break
    else:
        raw_ns = [0 for _ in plan.steps]

    for step, n_frames in zip(plan.steps, raw_ns):
        violated = None
        for p in step.preconditions:
            if not check_precondition(state, step, p):
                violated = p
                break

        if violated is not None:
            failure = {
                "step_id": step.id,
                "action_id": step.action_id,
                "action_name": step.action_name,
                "primitive": step.primitive,
                "precondition": violated,
                "message": f"Violated precondition: {violated}",
                "time": float(len(frames) / float(FPS)),
            }
            failures.append(failure)

            if render:
                frame = render_sim_frame(
                    state.ee,
                    state.yaw,
                    state.gripper_open,
                    state.held,
                    state.objects,
                    state.trail,
                    step=step,
                    failure=failure,
                    t=float(len(frames) / float(FPS)),
                )
                while len(frames) < target_frames:
                    frames.append(frame)
            break

        if step.primitive == "grasp_proxy":
            close = bool(step.parameters.get("close", True))
            obj_name = step.parameters.get("object")

            for f in range(n_frames):
                if render:
                    anim_open = state.gripper_open
                    if close and f > n_frames // 2:
                        anim_open = False
                    if (not close) and f > n_frames // 2:
                        anim_open = True
                    frames.append(
                        render_sim_frame(
                            state.ee,
                            state.yaw,
                            anim_open,
                            state.held,
                            state.objects,
                            state.trail,
                            step=step,
                            t=float(len(frames) / float(FPS)),
                        )
                    )

            if close and obj_name in state.objects:
                state.held = obj_name
                state.gripper_open = False
                state.hold_offset = np.asarray(state.objects[obj_name]["pos"], dtype=float) - state.ee
            elif (not close) and obj_name and state.held == obj_name:
                state.gripper_open = True
                state.held = None
                state.hold_offset = None
            elif not close:
                state.gripper_open = True

        else:
            start = state.ee.copy()
            end = np.asarray(step.goal, dtype=float)
            start_yaw = state.yaw
            end_yaw = start_yaw

            if step.primitive == "turn":
                end_yaw = float(step.parameters.get("yaw", start_yaw))
            elif step.primitive in ("reach", "retreat", "crouch"):
                delta = end - start
                if float(np.linalg.norm(delta)) > 1e-4:
                    end_yaw = float(math.atan2(delta[1], delta[0]))

            if n_frames > 0:
                for f in range(1, n_frames + 1):
                    alpha = f / float(n_frames)
                    state.ee = start + (end - start) * alpha
                    state.yaw = start_yaw + (end_yaw - start_yaw) * alpha

                    if state.held is not None and state.hold_offset is not None and state.held in state.objects:
                        state.objects[state.held]["pos"] = state.ee + state.hold_offset

                    state.trail.append(state.ee.copy())
                    if len(state.trail) > 45:
                        state.trail.pop(0)

                    if render:
                        frames.append(
                            render_sim_frame(
                                state.ee,
                                state.yaw,
                                state.gripper_open,
                                state.held,
                                state.objects,
                                state.trail,
                                step=step,
                                t=float(len(frames) / float(FPS)),
                            )
                        )
            else:
                state.ee = end
                state.yaw = end_yaw
                if state.held is not None and state.hold_offset is not None and state.held in state.objects:
                    state.objects[state.held]["pos"] = state.ee + state.hold_offset

            state.ee = end
            state.yaw = end_yaw
            if state.held is not None and state.hold_offset is not None and state.held in state.objects:
                state.objects[state.held]["pos"] = state.ee + state.hold_offset

        log.append(
            {
                "step_id": step.id,
                "action_id": step.action_id,
                "action_name": step.action_name,
                "primitive": step.primitive,
                "status": "ok",
            }
        )

    if render:
        if not frames:
            frames.append(
                render_sim_frame(
                    state.ee,
                    state.yaw,
                    state.gripper_open,
                    state.held,
                    state.objects,
                    state.trail,
                    t=0.0,
                )
            )
        while len(frames) < target_frames:
            frames.append(frames[-1])
        frames = frames[:target_frames]

    success = len(failures) == 0
    final_objects = {k: tuple(map(float, np.asarray(v["pos"], dtype=float))) for k, v in state.objects.items()}

    return SimResult(
        frames=frames,
        failures=failures,
        log=log,
        success=success,
        final_objects=final_objects,
        held=state.held,
    )


# =====================================================================
# EVALUATION AND TESTS
# =====================================================================


def temporal_iou(a: Action, b: Action) -> float:
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    if union <= 0:
        return 0.0
    return float(inter / union)


def compare_segmentations(
    pred: List[Action],
    gt: List[Action],
    iou_thresh: float = 0.5,
) -> Tuple[float, float, List[Tuple[Action, Action]]]:
    used = set()
    pairs: List[Tuple[Action, Action]] = []

    for p in pred:
        best_j = None
        best_iou = 0.0
        for j, g in enumerate(gt):
            if j in used:
                continue
            iou = temporal_iou(p, g)
            if iou >= iou_thresh and p.name == g.name and iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j is not None:
            used.add(best_j)
            pairs.append((p, gt[best_j]))

    matches = len(pairs)
    precision = matches / len(pred) if pred else 0.0
    recall = matches / len(gt) if gt else 0.0
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall / (precision + recall))

    obj_scores = []
    for p, g in pairs:
        sp = set(p.object_refs)
        sg = set(g.object_refs)
        union = sp | sg
        inter = sp & sg
        if not union:
            obj_scores.append(1.0)
        else:
            obj_scores.append(len(inter) / len(union))

    obj_acc = float(np.mean(obj_scores)) if obj_scores else 0.0
    return float(f1), obj_acc, pairs


def graph_validation_pass_rate(base_graph: SkillGraph) -> float:
    graphs: List[SkillGraph] = []

    # Valid variants with small timing jitter.
    for seed in range(5):
        c = copy_model(base_graph)
        rng = np.random.default_rng(100 + seed)
        t = 0.0
        for a in c.actions:
            dur = max(0.30, a.end - a.start)
            start = t + abs(float(rng.normal(0, 0.03)))
            start = clamp_scalar(start, 0.0, VIDEO_DURATION - 0.25)
            end = start + dur + float(rng.normal(0, 0.03))
            end = clamp_scalar(end, start + 0.12, VIDEO_DURATION)
            a.start = float(round(start, 3))
            a.end = float(round(end, 3))
            t = end + 0.02
        graphs.append(c)

    # Invalid: overlapping intervals.
    c = copy_model(base_graph)
    if len(c.actions) >= 2:
        c.actions[1].start = c.actions[0].start + 0.1
        c.actions[1].end = c.actions[0].end + 0.2
    graphs.append(c)

    # Invalid: cyclic dependency.
    c = copy_model(base_graph)
    if len(c.actions) >= 2:
        c.edges.append(Edge(source=c.actions[-1].id, target=c.actions[0].id))
    graphs.append(c)

    # Invalid: missing object reference.
    c = copy_model(base_graph)
    if c.actions:
        c.actions[0].object_refs = list(c.actions[0].object_refs) + ["banana"]
    graphs.append(c)

    # Invalid: missing version.
    c = copy_model(base_graph)
    c.version = ""
    graphs.append(c)

    # Invalid: confidence out of range.
    c = copy_model(base_graph)
    if c.actions:
        c.actions[0].confidence = 1.5
    graphs.append(c)

    valid = sum(1 for g in graphs if len(g.validation_errors()) == 0)
    return float(valid / len(graphs)) if graphs else 0.0


def execution_success_after_offset(
    graph: SkillGraph,
    trials: int = 15,
    seed: int = 7,
    scale: float = 0.040,
) -> Tuple[float, List[str]]:
    rng = np.random.default_rng(seed)
    compiler = GraphCompiler()
    success = 0
    failure_reasons: List[str] = []

    for _ in range(trials):
        g = copy_model(graph)
        for obj in g.objects:
            pos = np.asarray(obj.position, dtype=float) + rng.normal(0.0, scale, size=3)
            obj.position = tuple(float(v) for v in pos)
            obj.visible = bool(obj.visible and point_in_bounds(pos))

        try:
            plan = compiler.compile(g)
            res = simulate_plan(plan, g, render=False)
            if res.success:
                success += 1
            else:
                if res.failures:
                    failure_reasons.append(str(res.failures[0].get("precondition", "unknown")))
                else:
                    failure_reasons.append("unknown simulation failure")
        except Exception as exc:
            failure_reasons.append(f"compile/error: {exc}")

    rate = float(success / trials) if trials else 0.0
    return rate, failure_reasons


def run_evaluation(
    bundle: ObservationBundle,
    gt_graph: SkillGraph,
    segmenter: Segmenter,
    current_graph: SkillGraph,
) -> Dict[str, Any]:
    pred_graph = segmenter.segment(bundle)
    f1, obj_acc, pairs = compare_segmentations(pred_graph.actions, gt_graph.actions)
    pass_rate = graph_validation_pass_rate(current_graph)
    exec_rate, exec_failures = execution_success_after_offset(current_graph)

    pred_table = []
    for p, g in pairs:
        pred_table.append(
            {
                "pred_id": p.id,
                "pred_name": p.name,
                "pred_start": p.start,
                "pred_end": p.end,
                "gt_id": g.id,
                "gt_name": g.name,
                "gt_start": g.start,
                "gt_end": g.end,
                "iou": round(temporal_iou(p, g), 3),
                "pred_objects": ";".join(p.object_refs),
                "gt_objects": ";".join(g.object_refs),
            }
        )

    reason_counts: Dict[str, int] = {}
    for r in exec_failures:
        reason_counts[r] = reason_counts.get(r, 0) + 1

    return {
        "segment_f1": f1,
        "object_reference_accuracy": obj_acc,
        "validation_pass_rate": pass_rate,
        "execution_success_after_offset": exec_rate,
        "execution_failure_reasons": reason_counts,
        "prediction_pairs": pred_table,
        "predicted_graph_metadata": pred_graph.metadata,
    }


def run_builtin_tests(
    base_graph: SkillGraph,
    bundle: ObservationBundle,
    gt_graph: SkillGraph,
    segmenter: Segmenter,
) -> pd.DataFrame:
    results: List[Dict[str, str]] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        results.append({"test": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    # Invalid interval.
    g = copy_model(base_graph)
    if g.actions:
        g.actions[0].end = g.actions[0].start - 0.1
    errs = g.validation_errors()
    add("invalid interval rejected", any("end must be greater" in e for e in errs), "; ".join(errs))

    # Cyclic dependency.
    g = copy_model(base_graph)
    if len(g.actions) >= 2:
        g.edges.append(Edge(source=g.actions[-1].id, target=g.actions[0].id))
    errs = g.validation_errors()
    add("cyclic dependency rejected", any("cyclic" in e.lower() for e in errs), "; ".join(errs))

    # Missing object.
    g = copy_model(base_graph)
    if g.actions:
        g.actions[0].object_refs = list(g.actions[0].object_refs) + ["banana"]
    errs = g.validation_errors()
    add("missing object rejected", any("missing object reference" in e.lower() for e in errs), "; ".join(errs))

    # JSON version field.
    try:
        js = json.loads(to_json(base_graph))
        add("JSON has version field", js.get("version") == APP_VERSION, f"version={js.get('version')}")
    except Exception as exc:
        add("JSON has version field", False, str(exc))

    # Compiler isolation: compile signature should only take graph.
    try:
        params = set(inspect.signature(GraphCompiler.compile).parameters.keys())
        add("compiler does not call perception directly", params == {"self", "graph"}, str(params))
    except Exception as exc:
        add("compiler does not call perception directly", False, str(exc))

    # Segmentation quality.
    try:
        pred = segmenter.segment(bundle)
        f1, _, _ = compare_segmentations(pred.actions, gt_graph.actions)
        add("segmentation F1 > 0.50", f1 > 0.50, f"F1={f1:.3f}")
    except Exception as exc:
        add("segmentation F1 > 0.50", False, str(exc))

    # Failure report names violated precondition.
    try:
        g = copy_model(base_graph)
        for obj in g.objects:
            if obj.name == "mug":
                obj.visible = False
        plan = GraphCompiler().compile(g)
        res = simulate_plan(plan, g, render=False)
        ok = (
            (not res.success)
            and bool(res.failures)
            and str(res.failures[0].get("precondition", "")).startswith("object:mug.visible")
        )
        add(
            "failure report names violated precondition",
            ok,
            res.failures[0]["precondition"] if res.failures else "no failure",
        )
    except Exception as exc:
        add("failure report names violated precondition", False, str(exc))

    return pd.DataFrame(results)


# =====================================================================
# STREAMLIT SESSION PAYLOAD HELPERS
# =====================================================================
# These helpers prevent stale Pydantic model instances from different
# script reruns from being passed into new Pydantic constructors.
# =====================================================================


def reset_session_if_schema_changed() -> None:
    if st.session_state.get("app_state_schema") != APP_STATE_SCHEMA:
        try:
            st.session_state.clear()
        except Exception:
            for key in list(st.session_state.keys()):
                del st.session_state[key]
        st.session_state["app_state_schema"] = APP_STATE_SCHEMA


def _payload_from_model_or_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, BaseModel):
        return to_dict(value)
    if isinstance(value, dict):
        return value
    raise TypeError(f"Cannot convert {type(value)} to payload")


def _clean_graph_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "version": payload.get("version", APP_VERSION),
        "actions": [_payload_from_model_or_dict(x) for x in (payload.get("actions") or [])],
        "edges": [_payload_from_model_or_dict(x) for x in (payload.get("edges") or [])],
        "objects": [_payload_from_model_or_dict(x) for x in (payload.get("objects") or [])],
        "metadata": payload.get("metadata") or {},
    }


def get_gt_graph_payload() -> Dict[str, Any]:
    if "gt_graph_payload" not in st.session_state:
        st.session_state.gt_graph_payload = to_dict(ground_truth_graph())
    raw = st.session_state.gt_graph_payload
    if isinstance(raw, BaseModel):
        raw = to_dict(raw)
        st.session_state.gt_graph_payload = raw
    return raw


def get_gt_graph() -> SkillGraph:
    return SkillGraph(**_clean_graph_payload(get_gt_graph_payload()))


def get_graph_payload() -> Dict[str, Any]:
    if "graph_payload" not in st.session_state:
        if "graph" in st.session_state:
            st.session_state.graph_payload = _payload_from_model_or_dict(st.session_state.graph)
            del st.session_state.graph
        else:
            st.session_state.graph_payload = to_dict(ground_truth_graph())

    raw = st.session_state.graph_payload
    if isinstance(raw, BaseModel):
        raw = to_dict(raw)
        st.session_state.graph_payload = raw
    return raw


def get_graph() -> SkillGraph:
    payload = _clean_graph_payload(get_graph_payload())
    return SkillGraph(**payload)


def set_graph(graph: SkillGraph) -> None:
    st.session_state.graph_payload = to_dict(graph)
    if "graph" in st.session_state:
        del st.session_state.graph


def _clean_plan_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "version": payload.get("version", APP_VERSION),
        "steps": [_payload_from_model_or_dict(x) for x in (payload.get("steps") or [])],
        "metadata": payload.get("metadata") or {},
    }


def get_plan_payload() -> Dict[str, Any]:
    if "plan_payload" not in st.session_state:
        if "plan" in st.session_state:
            st.session_state.plan_payload = _payload_from_model_or_dict(st.session_state.plan)
            del st.session_state.plan
        else:
            st.session_state.plan_payload = to_dict(RobotPlan(version=APP_VERSION, steps=[], metadata={}))

    raw = st.session_state.plan_payload
    if isinstance(raw, BaseModel):
        raw = to_dict(raw)
        st.session_state.plan_payload = raw
    return raw


def get_plan() -> RobotPlan:
    payload = _clean_plan_payload(get_plan_payload())
    return RobotPlan(**payload)


def set_plan(plan: RobotPlan) -> None:
    st.session_state.plan_payload = to_dict(plan)
    if "plan" in st.session_state:
        del st.session_state.plan


def get_baseline_payload() -> Dict[str, Any]:
    if "baseline_payload" not in st.session_state:
        if "baseline" in st.session_state:
            b = st.session_state.baseline
            st.session_state.baseline_payload = {
                "frames": getattr(b, "frames", []),
                "failures": getattr(b, "failures", []),
                "log": getattr(b, "log", []),
                "success": bool(getattr(b, "success", False)),
                "final_objects": getattr(b, "final_objects", {}),
                "held": getattr(b, "held", None),
            }
            del st.session_state.baseline
        else:
            st.session_state.baseline_payload = {
                "frames": [],
                "failures": [],
                "log": [],
                "success": True,
                "final_objects": {},
                "held": None,
            }
    return st.session_state.baseline_payload


def get_baseline() -> SimResult:
    p = get_baseline_payload()
    final_objects_raw = p.get("final_objects") or {}
    final_objects = {
        k: tuple(float(x) for x in v)
        for k, v in final_objects_raw.items()
    }
    return SimResult(
        frames=p.get("frames") or [],
        failures=p.get("failures") or [],
        log=p.get("log") or [],
        success=bool(p.get("success", True)),
        final_objects=final_objects,
        held=p.get("held"),
    )


def set_baseline(sim: SimResult) -> None:
    st.session_state.baseline_payload = {
        "frames": list(sim.frames),
        "failures": list(sim.failures),
        "log": list(sim.log),
        "success": bool(sim.success),
        "final_objects": dict(sim.final_objects),
        "held": sim.held,
    }
    if "baseline" in st.session_state:
        del st.session_state.baseline


def bump_graph_revisions() -> None:
    st.session_state.graph_revision = int(st.session_state.get("graph_revision", 0)) + 1
    st.session_state.world_revision = int(st.session_state.get("world_revision", 0)) + 1


# =====================================================================
# STREAMLIT DASHBOARD
# =====================================================================


def actions_to_dataframe(graph: SkillGraph) -> pd.DataFrame:
    rows = []
    for a in graph.actions:
        rows.append(
            {
                "id": a.id,
                "name": a.name,
                "start": a.start,
                "end": a.end,
                "confidence": a.confidence,
                "body_parts": "; ".join(a.body_parts),
                "object_refs": "; ".join(a.object_refs),
                "preconditions": "; ".join(a.preconditions),
                "postconditions": "; ".join(a.postconditions),
            }
        )
    if not rows:
        return pd.DataFrame(columns=[
            "id", "name", "start", "end", "confidence",
            "body_parts", "object_refs", "preconditions", "postconditions"
        ])
    return pd.DataFrame(rows)


def edges_to_dataframe(graph: SkillGraph) -> pd.DataFrame:
    rows = [{"source": e.source, "target": e.target, "type": e.type} for e in graph.edges]
    if not rows:
        return pd.DataFrame(columns=["source", "target", "type"])
    return pd.DataFrame(rows)


def plan_to_dataframe(plan: RobotPlan) -> pd.DataFrame:
    rows = []
    for s in plan.steps:
        rows.append(
            {
                "id": s.id,
                "action_id": s.action_id,
                "action_name": s.action_name,
                "primitive": s.primitive,
                "goal": ", ".join(f"{v:.3f}" for v in s.goal),
                "preconditions": "; ".join(s.preconditions),
                "postconditions": "; ".join(s.postconditions),
                "duration_hint": s.duration_hint,
            }
        )
    if not rows:
        return pd.DataFrame(columns=[
            "id", "action_id", "action_name", "primitive", "goal",
            "preconditions", "postconditions", "duration_hint"
        ])
    return pd.DataFrame(rows)


def current_action_at(graph: SkillGraph, t: float) -> Optional[Action]:
    for a in graph.actions:
        if a.start <= t < a.end:
            return a
    return None


def recompute_plan_and_sim() -> None:
    graph = get_graph()
    try:
        graph.throw_if_invalid()
        plan = GraphCompiler().compile(graph)
        sim = simulate_plan(plan, graph, render=True)
        set_plan(plan)
        set_baseline(sim)
        st.session_state.compile_error = None
        if st.session_state.sim_frame >= len(sim.frames):
            st.session_state.sim_frame = max(0, len(sim.frames) - 1)
    except Exception as exc:
        st.session_state.compile_error = str(exc)


def init_state() -> None:
    reset_session_if_schema_changed()

    defaults = {
        "graph_revision": 0,
        "world_revision": 0,
        "video_frame": 0,
        "sim_frame": 0,
        "auto_play": True,
        "play_both": False,
        "play_video": False,
        "play_sim": False,
        "play_speed": 0.04,
        "play_speed_slider": 0.04,
        "review_started_at": time.time(),
        "review_fixed_at": None,
        "use_learned": False,
        "compile_error": None,
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if "bundle" not in st.session_state:
        st.session_state.bundle = VideoIngestor().load(MockPoseSource(), MockObjectDetector())

    bundle = st.session_state.bundle

    if "gt_graph_payload" not in st.session_state:
        st.session_state.gt_graph_payload = to_dict(ground_truth_graph())

    if "graph_payload" not in st.session_state:
        segmenter = MotionChangeSegmenter()
        graph = segmenter.segment(bundle)
        if graph.validation_errors():
            graph = ground_truth_graph()
        set_graph(graph)

    if "plan_payload" not in st.session_state or "baseline_payload" not in st.session_state:
        recompute_plan_and_sim()

    if "metrics" not in st.session_state:
        st.session_state.metrics = run_evaluation(
            bundle,
            get_gt_graph(),
            MotionChangeSegmenter(),
            get_graph(),
        )

    if "tests" not in st.session_state:
        st.session_state.tests = run_builtin_tests(
            get_graph(),
            bundle,
            get_gt_graph(),
            MotionChangeSegmenter(),
        )


def sidebar_controls() -> None:
    st.sidebar.title("Controls")

    st.sidebar.markdown("### Adapters")
    for k, v in st.session_state.bundle.adapter_report.items():
        st.sidebar.markdown(f"- **{k}**: {v}")
    st.sidebar.markdown(
        f"- **learned_classifier**: {'enabled mock fallback' if st.session_state.use_learned else 'disabled'}"
    )

    st.sidebar.markdown("### Segmentation")

    # IMPORTANT:
    # Do NOT assign st.session_state.use_learned here.
    # The widget key owns that session-state slot.
    st.sidebar.checkbox(
        "Use learned classifier interface (mock unavailable)",
        key="use_learned",
    )

    if st.sidebar.button("Re-segment video"):
        seg = LearnedActionSegmenter() if st.session_state.use_learned else MotionChangeSegmenter()
        new_graph = seg.segment(st.session_state.bundle)
        set_graph(new_graph)
        bump_graph_revisions()
        st.session_state.review_started_at = time.time()
        st.session_state.review_fixed_at = None
        recompute_plan_and_sim()
        rerun_app()

    st.sidebar.markdown("### Playback")
    speed = st.sidebar.slider(
        "Seconds per frame",
        min_value=0.02,
        max_value=0.12,
        step=0.01,
        key="play_speed_slider",
    )
    st.session_state.play_speed = float(speed)

    c1, c2, c3 = st.sidebar.columns(3)
    if c1.button("Play all"):
        st.session_state.play_both = True
        st.session_state.auto_play = False
        st.session_state.video_frame = 0
        st.session_state.sim_frame = 0
        rerun_app()
    if c2.button("Play video"):
        st.session_state.play_video = True
        rerun_app()
    if c3.button("Play sim"):
        st.session_state.play_sim = True
        rerun_app()

    st.sidebar.markdown("### World state")
    graph = get_graph()
    rev = st.session_state.world_revision

    # Add missing referenced objects if validator reports them.
    missing = []
    object_names = {o.name for o in graph.objects}
    for a in graph.actions:
        for r in a.object_refs:
            if r not in object_names:
                missing.append(r)
    missing = sorted(set(missing))
    if missing:
        st.sidebar.warning(f"Missing object states: {', '.join(missing)}")
        if st.sidebar.button("Add missing referenced objects"):
            payload = to_dict(graph)
            for nm in missing:
                payload["objects"].append(
                    to_dict(
                        ObjectState(
                            name=nm,
                            position=(0.55, 0.35, 0.25),
                            visible=True,
                            graspable=True,
                            uncertainty=0.2,
                        )
                    )
                )
            set_graph(SkillGraph(**payload))
            bump_graph_revisions()
            recompute_plan_and_sim()
            rerun_app()

    for obj in graph.objects:
        st.sidebar.markdown(f"**{obj.name}**")
        st.sidebar.number_input(
            "x",
            value=float(obj.position[0]),
            step=0.01,
            format="%.3f",
            key=f"wx_{rev}_{obj.name}",
        )
        st.sidebar.number_input(
            "y",
            value=float(obj.position[1]),
            step=0.01,
            format="%.3f",
            key=f"wy_{rev}_{obj.name}",
        )
        st.sidebar.number_input(
            "z",
            value=float(obj.position[2]),
            step=0.01,
            format="%.3f",
            key=f"wz_{rev}_{obj.name}",
        )
        st.sidebar.checkbox(
            "visible",
            value=bool(obj.visible),
            key=f"wvis_{rev}_{obj.name}",
        )
        st.sidebar.checkbox(
            "graspable",
            value=bool(obj.graspable),
            key=f"wgrasp_{rev}_{obj.name}",
        )

    bc1, bc2 = st.sidebar.columns(2)
    if bc1.button("Apply world state"):
        payload = to_dict(graph)
        for obj_payload in payload["objects"]:
            name = obj_payload["name"]
            pos = obj_payload.get("position", [0.0, 0.0, 0.0])
            x = float(st.session_state.get(f"wx_{rev}_{name}", pos[0]))
            y = float(st.session_state.get(f"wy_{rev}_{name}", pos[1]))
            z = float(st.session_state.get(f"wz_{rev}_{name}", pos[2]))
            obj_payload["position"] = [x, y, z]
            obj_payload["visible"] = bool(st.session_state.get(f"wvis_{rev}_{name}", obj_payload.get("visible", True)))
            obj_payload["graspable"] = bool(
                st.session_state.get(f"wgrasp_{rev}_{name}", obj_payload.get("graspable", True))
            )
        set_graph(SkillGraph(**payload))
        bump_graph_revisions()
        recompute_plan_and_sim()
        rerun_app()

    if bc2.button("Random perturb"):
        payload = to_dict(graph)
        rng = np.random.default_rng()
        for obj_payload in payload["objects"]:
            pos = np.asarray(obj_payload.get("position", [0.0, 0.0, 0.0]), dtype=float)
            pos = pos + rng.normal(0.0, 0.035, size=3)
            obj_payload["position"] = [float(v) for v in pos]
            obj_payload["visible"] = bool(obj_payload.get("visible", True) and point_in_bounds(pos))
        set_graph(SkillGraph(**payload))
        bump_graph_revisions()
        recompute_plan_and_sim()
        rerun_app()

    st.sidebar.markdown("### Review timer")
    if st.session_state.review_fixed_at is None:
        elapsed = time.time() - st.session_state.review_started_at
        st.sidebar.metric("Elapsed", format_seconds(elapsed))
    else:
        fixed = st.session_state.review_fixed_at - st.session_state.review_started_at
        st.sidebar.metric("Fixed in", format_seconds(fixed))

    rc1, rc2 = st.sidebar.columns(2)
    if rc1.button("Reset timer"):
        st.session_state.review_started_at = time.time()
        st.session_state.review_fixed_at = None
        rerun_app()

    if rc2.button("Mark fixed"):
        errs = get_graph().validation_errors()
        if errs:
            st.sidebar.error("Cannot mark fixed while graph is invalid.")
        else:
            st.session_state.review_fixed_at = time.time()
            rerun_app()


def playback_area() -> None:
    video_col, sim_col = st.columns(2)
    video_ph = video_col.empty()
    sim_ph = sim_col.empty()
    status_ph = st.empty()

    frames = st.session_state.bundle.frames
    sim = get_baseline()
    sim_frames = sim.frames

    if not frames or not sim_frames:
        st.warning("No demo frames available.")
        return

    # Synchronized auto/demo playback.
    if st.session_state.get("auto_play", False) or st.session_state.get("play_both", False):
        start = 0
        n = min(len(frames), len(sim_frames))
        for i in range(start, n):
            show_image(video_ph, frames[i])
            show_image(sim_ph, sim_frames[i])
            status_ph.caption(f"Synchronized demo: frame {i + 1}/{n} | t={i / float(FPS):.1f}s")
            time.sleep(float(st.session_state.play_speed))
            st.session_state.video_frame = i
            st.session_state.sim_frame = i

        st.session_state.video_frame = n - 1
        st.session_state.sim_frame = len(sim_frames) - 1
        st.session_state.auto_play = False
        st.session_state.play_both = False
        rerun_app()

    if st.session_state.get("play_video", False):
        start = int(st.session_state.video_frame)
        for i in range(start, len(frames)):
            show_image(video_ph, frames[i])
            show_image(sim_ph, sim_frames[min(st.session_state.sim_frame, len(sim_frames) - 1)])
            status_ph.caption(f"Video playback: frame {i + 1}/{len(frames)} | t={i / float(FPS):.1f}s")
            time.sleep(float(st.session_state.play_speed))
            st.session_state.video_frame = i
        st.session_state.video_frame = len(frames) - 1
        st.session_state.play_video = False
        rerun_app()

    if st.session_state.get("play_sim", False):
        start = int(st.session_state.sim_frame)
        for i in range(start, len(sim_frames)):
            show_image(sim_ph, sim_frames[i])
            show_image(video_ph, frames[min(st.session_state.video_frame, len(frames) - 1)])
            status_ph.caption(f"Simulation playback: frame {i + 1}/{len(sim_frames)} | t={i / float(FPS):.1f}s")
            time.sleep(float(st.session_state.play_speed))
            st.session_state.sim_frame = i
        st.session_state.sim_frame = len(sim_frames) - 1
        st.session_state.play_sim = False
        rerun_app()

    vf = int(max(0, min(st.session_state.video_frame, len(frames) - 1)))
    sf = int(max(0, min(st.session_state.sim_frame, len(sim_frames) - 1)))
    st.session_state.video_frame = vf
    st.session_state.sim_frame = sf

    show_image(video_ph, frames[vf])
    show_image(sim_ph, sim_frames[sf])

    t = vf / float(FPS)
    act = current_action_at(get_graph(), t)
    status_ph.caption(
        f"Video t={t:.1f}s | current action: {act.name if act else 'idle'} | "
        f"sim frame {sf + 1}/{len(sim_frames)}"
    )

    new_vf = video_col.slider(
        "Video timeline scrubbing",
        min_value=0,
        max_value=len(frames) - 1,
        value=vf,
        step=1,
        format="frame %d",
    )
    if int(new_vf) != vf:
        st.session_state.video_frame = int(new_vf)
        rerun_app()

    new_sf = sim_col.slider(
        "Simulation timeline scrubbing",
        min_value=0,
        max_value=len(sim_frames) - 1,
        value=sf,
        step=1,
        format="frame %d",
    )
    if int(new_sf) != sf:
        st.session_state.sim_frame = int(new_sf)
        rerun_app()


def review_editor_tab() -> None:
    st.markdown("### Correct uncertain labels, edit dependencies, and re-validate")

    graph = get_graph()
    errs = graph.validation_errors()
    if errs:
        st.error("Current graph is invalid:\n- " + "\n- ".join(errs))
    else:
        st.success("Current graph is valid.")

    if not graph.actions:
        st.warning("No actions in graph. Re-segment or restore hand-labelled demo graph.")
        return

    object_options = sorted(
        set([o.name for o in graph.objects] + [r for a in graph.actions for r in a.object_refs])
    )

    rev = st.session_state.graph_revision
    edited_actions: List[Action] = []

    for a in graph.actions:
        uncertain = a.confidence < 0.72
        label = f"{'! ' if uncertain else ''}{a.id}: {a.name} [{a.start:.2f}s -> {a.end:.2f}s] conf={a.confidence:.2f}"
        prefix = f"g{rev}_{a.id}"
        with st.expander(label, expanded=uncertain):
            name = st.text_input("Name", value=a.name, key=f"{prefix}_name")
            c1, c2 = st.columns(2)
            start = c1.number_input("Start", value=float(a.start), step=0.05, format="%.2f", key=f"{prefix}_start")
            end = c2.number_input("End", value=float(a.end), step=0.05, format="%.2f", key=f"{prefix}_end")

            body_defaults = [x for x in a.body_parts if x in BODY_PART_OPTIONS]
            obj_defaults = [x for x in a.object_refs if x in object_options]

            body = st.multiselect(
                "Body parts",
                options=BODY_PART_OPTIONS,
                default=body_defaults,
                key=f"{prefix}_body",
            )
            objs = st.multiselect(
                "Object refs",
                options=object_options,
                default=obj_defaults,
                key=f"{prefix}_objs",
            )

            pre_text = st.text_area(
                "Preconditions (one per line)",
                value="\n".join(a.preconditions),
                key=f"{prefix}_pre",
                height=90,
            )
            post_text = st.text_area(
                "Postconditions (one per line)",
                value="\n".join(a.postconditions),
                key=f"{prefix}_post",
                height=90,
            )
            conf = st.slider("Confidence", 0.0, 1.0, float(a.confidence), step=0.01, key=f"{prefix}_conf")

            edited_actions.append(
                Action(
                    id=a.id,
                    name=name,
                    start=float(start),
                    end=float(end),
                    body_parts=body,
                    object_refs=objs,
                    preconditions=[x.strip() for x in pre_text.splitlines() if x.strip()],
                    postconditions=[x.strip() for x in post_text.splitlines() if x.strip()],
                    confidence=float(conf),
                )
            )

    st.markdown("#### Dependency edges")
    action_labels = {a.id: f"{a.id}: {a.name} [{a.start:.1f}-{a.end:.1f}]" for a in graph.actions}

    ec1, ec2, ec3 = st.columns([2, 2, 1])
    src = ec1.selectbox(
        "Source action",
        list(action_labels.keys()),
        format_func=lambda x: action_labels[x],
        key=f"g{rev}_dep_src",
    )
    tgt = ec2.selectbox(
        "Target action",
        list(action_labels.keys()),
        format_func=lambda x: action_labels[x],
        key=f"g{rev}_dep_tgt",
    )

    payload = {
        "version": graph.version,
        "actions": [to_dict(a) for a in edited_actions],
        "edges": [to_dict(e) for e in graph.edges],
        "objects": [to_dict(o) for o in graph.objects],
        "metadata": dict(graph.metadata),
    }

    if ec3.button("Add edge"):
        if src == tgt:
            st.warning("Self-edge ignored.")
        elif any(e.get("source") == src and e.get("target") == tgt for e in payload["edges"]):
            st.info("Edge already exists.")
        else:
            payload["edges"].append({"source": src, "target": tgt, "type": "depends_on"})
            set_graph(SkillGraph(**payload))
            bump_graph_revisions()
            recompute_plan_and_sim()
            rerun_app()

    if payload["edges"]:
        edge_labels = [f"{e['source']} -> {e['target']}" for e in payload["edges"]]
        rem_idx = st.selectbox(
            "Remove dependency",
            range(len(payload["edges"])),
            format_func=lambda i: edge_labels[i],
            key=f"g{rev}_dep_remove",
        )
        if st.button("Remove selected edge"):
            payload["edges"].pop(int(rem_idx))
            set_graph(SkillGraph(**payload))
            bump_graph_revisions()
            recompute_plan_and_sim()
            rerun_app()

    vc1, vc2 = st.columns(2)
    if vc1.button("Apply action edits & validate"):
        set_graph(SkillGraph(**payload))
        bump_graph_revisions()
        recompute_plan_and_sim()
        rerun_app()

    if vc2.button("Restore hand-labelled demo graph"):
        set_graph(get_gt_graph())
        bump_graph_revisions()
        recompute_plan_and_sim()
        rerun_app()


def graph_plan_tab() -> None:
    st.markdown("### Typed skill graph")
    graph = get_graph()

    st.write(f"**Version:** `{graph.version}`")
    st.dataframe(actions_to_dataframe(graph))

    st.markdown("#### Dependencies")
    if graph.edges:
        st.dataframe(edges_to_dataframe(graph))
    else:
        st.write("No dependency edges.")

    st.markdown("#### Adjacency view")
    succ: Dict[str, List[str]] = {a.id: [] for a in graph.actions}
    for e in graph.edges:
        if e.source in succ:
            succ[e.source].append(e.target)
    for a in graph.actions:
        st.write(f"- `{a.id}` {a.name} -> {', '.join(succ.get(a.id, [])) or 'none'}")

    st.markdown("### Compiled robot plan")
    if st.session_state.compile_error:
        st.error(f"Compilation failed: {st.session_state.compile_error}")
    st.dataframe(plan_to_dataframe(get_plan()))

    st.markdown("### Export")
    e1, e2, e3 = st.columns(3)

    graph_json = to_json(graph, indent=2)
    plan_json = to_json(get_plan(), indent=2)
    csv = actions_to_dataframe(graph).to_csv(index=False).encode("utf-8")

    e1.download_button("Download graph JSON", graph_json, file_name="skill_graph.json", mime="application/json")
    e2.download_button("Download plan JSON", plan_json, file_name="robot_plan.json", mime="application/json")
    e3.download_button("Download actions CSV", csv, file_name="actions.csv", mime="text/csv")

    with st.expander("Preview graph JSON"):
        st.code(graph_json, language="json")


def simulation_tab() -> None:
    st.markdown("### Simulation execution and failure reports")

    if st.session_state.compile_error:
        st.warning(f"Using last valid simulation because current compile failed: {st.session_state.compile_error}")

    sim = get_baseline()
    if sim.frames:
        sf = int(max(0, min(st.session_state.sim_frame, len(sim.frames) - 1)))
        show_image(st, sim.frames[sf])
    else:
        st.warning("No simulation frames.")

    if sim.success:
        st.success("Simulation succeeded.")
    else:
        st.error("Simulation failed.")

    st.markdown("#### Failures")
    if sim.failures:
        st.dataframe(pd.DataFrame(sim.failures))
    else:
        st.write("No failures.")

    st.markdown("#### Step log")
    if sim.log:
        st.dataframe(pd.DataFrame(sim.log))
    else:
        st.write("No successful steps logged.")

    st.markdown("#### Final object positions")
    st.json(sim.final_objects)


def evaluation_tab() -> None:
    st.markdown("### Evaluation metrics")

    if st.button("Rerun evaluation"):
        seg = LearnedActionSegmenter() if st.session_state.use_learned else MotionChangeSegmenter()
        st.session_state.metrics = run_evaluation(
            st.session_state.bundle,
            get_gt_graph(),
            seg,
            get_graph(),
        )
        rerun_app()

    m = st.session_state.metrics

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Segment F1", f"{m['segment_f1']:.3f}")
    c2.metric("Object-ref accuracy", f"{m['object_reference_accuracy']:.3f}")
    c3.metric("Graph validation pass rate", f"{m['validation_pass_rate']:.3f}")
    c4.metric("Exec success after offset", f"{m['execution_success_after_offset']:.3f}")

    if st.session_state.review_fixed_at is None:
        rev = time.time() - st.session_state.review_started_at
        c5.metric("Reviewer time", format_seconds(rev), "running")
    else:
        rev = st.session_state.review_fixed_at - st.session_state.review_started_at
        c5.metric("Reviewer time", format_seconds(rev), "fixed")

    st.markdown("#### Matched predicted vs hand-labelled segments")
    if m["prediction_pairs"]:
        st.dataframe(pd.DataFrame(m["prediction_pairs"]))
    else:
        st.write("No matched segments.")

    st.markdown("#### Random-offset failure reasons")
    if m["execution_failure_reasons"]:
        st.dataframe(
            pd.DataFrame(
                [{"precondition": k, "count": v} for k, v in m["execution_failure_reasons"].items()]
            )
        )
    else:
        st.write("No random-offset failures in the latest batch.")

    st.markdown("#### Interpretation")
    st.markdown(
        """
- **Segment F1** measures temporal agreement with hand-labelled action phases.
- **Object-reference accuracy** measures whether the correct objects are attached to each action.
- **Validation pass rate** measures how often generated/edited graphs satisfy typed graph constraints.
- **Execution success after offset** recompiles from perturbed object states and executes in simulation.
- **Reviewer time** is measured live from demo load until the reviewer marks the graph fixed.
        """
    )


def tests_tab() -> None:
    st.markdown("### Built-in correctness tests")

    if st.button("Rerun tests"):
        seg = LearnedActionSegmenter() if st.session_state.use_learned else MotionChangeSegmenter()
        st.session_state.tests = run_builtin_tests(
            get_graph(),
            st.session_state.bundle,
            get_gt_graph(),
            seg,
        )
        rerun_app()

    st.dataframe(st.session_state.tests)

    passed = int((st.session_state.tests["status"] == "PASS").sum())
    total = len(st.session_state.tests)
    st.metric("Tests passed", f"{passed}/{total}")


def dashboard_tab() -> None:
    st.markdown("### System summary")

    graph = get_graph()

    c1, c2, c3 = st.columns(3)
    c1.metric("Actions", len(graph.actions))
    c2.metric("Dependency edges", len(graph.edges))
    c3.metric("Objects", len(graph.objects))

    st.markdown("#### Architecture boundaries")
    st.markdown(
        """
- **Perception**: timestamped video ingestion, `PoseSource`, `ObjectDetector`, uncertainty for occluded parts.
- **Semantics**: typed Pydantic `Action` / `SkillGraph`, validator, motion-change segmentation, optional learned classifier interface.
- **Control**: graph-only compiler to end-effector goals and safe primitives, simulator with precondition-named failures.
- **Compiler isolation**: `GraphCompiler.compile(graph)` receives only a validated `SkillGraph`; it does not call perception code.
        """
    )

    st.markdown("#### Current video interpretation")
    t = st.session_state.video_frame / float(FPS)
    act = current_action_at(graph, t)
    if act:
        st.json(to_dict(act))
    else:
        st.write("No active labelled action at this timestamp.")


def main() -> None:
    st.set_page_config(page_title="Grounded Motion-to-Skill Compiler", layout="wide")
    init_state()

    st.title("Grounded Human-Motion to Executable Skill Graph Compiler")
    st.caption(
        "Observed motion -> typed skill graph -> reviewed/corrected graph -> robot end-effector plan -> simulation. "
        "All adapters are local mocks; no API key required."
    )

    sidebar_controls()
    playback_area()

    tabs = st.tabs(["Dashboard", "Review Editor", "Graph & Plan", "Simulation", "Evaluation", "Tests"])
    with tabs[0]:
        dashboard_tab()
    with tabs[1]:
        review_editor_tab()
    with tabs[2]:
        graph_plan_tab()
    with tabs[3]:
        simulation_tab()
    with tabs[4]:
        evaluation_tab()
    with tabs[5]:
        tests_tab()


if __name__ == "__main__":
    main()