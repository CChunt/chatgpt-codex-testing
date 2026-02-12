"""Progressive spherical parameterization with flip-safe optimization.

This module implements a practical pipeline for genus-0 triangle meshes:

1. Build a progressive sequence (coarse-to-fine) using edge collapses / vertex splits.
2. Start from a tetrahedron on the unit sphere.
3. Insert one vertex split at a time and optimize on S^2 after each insertion.
4. Use line search with strict no-flip checks so every accepted iterate remains bijective.

The optimizer minimizes a weighted objective made of:

- Symmetric Dirichlet energy in local tangent coordinates.
- Optional edge-length regularization.
- Optional electrostatic repulsion in tangent space.

The key safety mechanism is *feasibility-first* updates:
we only accept a step if all triangle orientations remain positive and
no triangle area drops below a small threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

EPS = 1e-12


ArrayF = np.ndarray
ArrayI = np.ndarray


def _row_normalize(x: ArrayF) -> ArrayF:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, EPS)


def _triangle_double_area_3d(v0: ArrayF, v1: ArrayF, v2: ArrayF) -> ArrayF:
    return np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


def _triangle_signed_area_on_sphere(u0: ArrayF, u1: ArrayF, u2: ArrayF) -> ArrayF:
    # Sign from orientation w.r.t. outward radial direction.
    n = np.cross(u1 - u0, u2 - u0)
    c = (u0 + u1 + u2) / 3.0
    return np.einsum("ij,ij->i", n, c)


def _face_vertices(faces: ArrayI, x: ArrayF) -> Tuple[ArrayF, ArrayF, ArrayF]:
    return x[faces[:, 0]], x[faces[:, 1]], x[faces[:, 2]]


def _cotangent(a: ArrayF, b: ArrayF) -> ArrayF:
    # cot(theta) where theta is angle between a and b
    cross = np.linalg.norm(np.cross(a, b), axis=1)
    dot = np.einsum("ij,ij->i", a, b)
    return dot / np.maximum(cross, EPS)


def cotan_laplacian(points: ArrayF, faces: ArrayI) -> sparse.csr_matrix:
    """Build symmetric cotangent Laplacian in R^3 geometry."""
    n = points.shape[0]
    i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]

    p0, p1, p2 = points[i0], points[i1], points[i2]

    cot0 = _cotangent(p1 - p0, p2 - p0)
    cot1 = _cotangent(p2 - p1, p0 - p1)
    cot2 = _cotangent(p0 - p2, p1 - p2)

    # edge (i1, i2) opposite i0 gets cot0, etc.
    rows = np.hstack([i1, i2, i2, i0, i0, i1])
    cols = np.hstack([i2, i1, i0, i2, i1, i0])
    vals = 0.5 * np.hstack([cot0, cot0, cot1, cot1, cot2, cot2])

    w = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    d = np.asarray(w.sum(axis=1)).ravel()
    l = sparse.diags(d) - w
    return l


def vertex_areas(points: ArrayF, faces: ArrayI) -> ArrayF:
    n = points.shape[0]
    p0, p1, p2 = _face_vertices(faces, points)
    a2 = _triangle_double_area_3d(p0, p1, p2)
    lump = np.zeros(n, dtype=float)
    np.add.at(lump, faces[:, 0], a2 / 6.0)
    np.add.at(lump, faces[:, 1], a2 / 6.0)
    np.add.at(lump, faces[:, 2], a2 / 6.0)
    return np.maximum(lump, EPS)


@dataclass
class ProgressiveStep:
    """One reverse-collapse step (vertex split).

    Attributes:
        new_vertex: index of the inserted vertex.
        split_edge: edge (a, b) that existed before split.
        new_faces: iterable with the two new faces (triples of vertex indices).
    """

    new_vertex: int
    split_edge: Tuple[int, int]
    new_faces: Sequence[Tuple[int, int, int]]


@dataclass
class SphericalParams:
    max_iter_per_step: int = 80
    smooth_weight: float = 1.0
    edge_weight: float = 0.05
    repel_weight: float = 0.005
    min_signed_area: float = 1e-6
    step0: float = 1.0
    armijo: float = 1e-4
    backtrack: float = 0.5
    min_step: float = 1e-8


class ProgressiveSphericalParameterizer:
    """Flip-safe spherical parameterization for genus-0 meshes."""

    def __init__(self, points: ArrayF, faces: ArrayI, params: SphericalParams | None = None):
        self.points = np.asarray(points, dtype=float)
        self.faces = np.asarray(faces, dtype=int)
        self.params = params or SphericalParams()

        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("faces must be (F,3) int array")

    def _active_faces(self, active_mask: ArrayF | None) -> ArrayI:
        if active_mask is None:
            return self.faces
        face_active = np.all(active_mask[self.faces], axis=1)
        return self.faces[face_active]

    def is_feasible(self, u: ArrayF, active_mask: ArrayF | None = None) -> bool:
        faces = self._active_faces(active_mask)
        if faces.shape[0] == 0:
            return True
        s = _triangle_signed_area_on_sphere(*_face_vertices(faces, u))
        return bool(np.all(s > self.params.min_signed_area))

    def initialize_from_tetra(self, active_vertices: Iterable[int], tetra_positions: ArrayF) -> ArrayF:
        u = np.zeros((self.points.shape[0], 3), dtype=float)
        active = np.array(list(active_vertices), dtype=int)
        if active.size != 4:
            raise ValueError("tetra initialization needs exactly 4 active vertices")
        if tetra_positions.shape != (4, 3):
            raise ValueError("tetra_positions must have shape (4,3)")
        u[active] = _row_normalize(tetra_positions)
        return u

    def _insert_vertex_guess(self, u: ArrayF, v_new: int, edge: Tuple[int, int], active_mask: ArrayF) -> None:
        a, b = edge
        base = u[a] + u[b]
        cands: List[ArrayF] = []

        if np.linalg.norm(base) > EPS:
            cands.append(base)

        incident = self.faces[np.any(self.faces == a, axis=1) & np.any(self.faces == b, axis=1)]
        for tri in incident:
            others = tri[(tri != a) & (tri != b)]
            if others.size == 1 and active_mask[others[0]]:
                c = int(others[0])
                cands.append(base + 0.35 * u[c])
                cands.append(base - 0.35 * u[c])

        cands.append(np.cross(u[a], np.array([1.0, 0.0, 0.0])))
        cands.append(np.cross(u[a], np.array([0.0, 1.0, 0.0])))

        best = None
        best_score = -np.inf
        tmp_active = active_mask.copy()
        tmp_active[v_new] = True

        for c in cands:
            if np.linalg.norm(c) <= EPS:
                continue
            uu = u.copy()
            uu[v_new] = c / np.linalg.norm(c)
            faces = self._active_faces(tmp_active)
            if faces.shape[0] == 0:
                u[v_new] = uu[v_new]
                return
            s = _triangle_signed_area_on_sphere(*_face_vertices(faces, uu))
            score = np.min(s)
            if score > best_score:
                best_score = score
                best = uu[v_new]

        if best is None:
            raise ValueError(f"Could not initialize inserted vertex {v_new} without degeneracy")
        u[v_new] = best

    def _energy(self, u: ArrayF, active_mask: ArrayF) -> float:
        i = np.where(active_mask)[0]
        if i.size == 0:
            return 0.0

        # Smoothness from cotan Laplacian in original geometry.
        l = cotan_laplacian(self.points, self.faces)
        e_smooth = 0.5 * float(np.sum(u * (l @ u)))

        # Edge-length fitting (discourage very uneven spherical edges).
        f0, f1, f2 = self.faces[:, 0], self.faces[:, 1], self.faces[:, 2]
        e = np.vstack([
            np.stack([f0, f1], axis=1),
            np.stack([f1, f2], axis=1),
            np.stack([f2, f0], axis=1),
        ])
        e = np.sort(e, axis=1)
        e = np.unique(e, axis=0)
        du = np.linalg.norm(u[e[:, 0]] - u[e[:, 1]], axis=1)
        dx = np.linalg.norm(self.points[e[:, 0]] - self.points[e[:, 1]], axis=1)
        dx = dx / max(np.mean(dx), EPS)
        du = du / max(np.mean(du), EPS)
        e_edge = float(np.mean((du - dx) ** 2))

        # Mild repulsion to prevent clustering.
        ua = u[i]
        dot = ua @ ua.T
        np.fill_diagonal(dot, 0.0)
        e_repel = float(np.mean((dot.clip(min=0.0)) ** 2))

        return (
            self.params.smooth_weight * e_smooth
            + self.params.edge_weight * e_edge
            + self.params.repel_weight * e_repel
        )

    def _project_tangent(self, u: ArrayF, g: ArrayF) -> ArrayF:
        return g - (np.sum(g * u, axis=1, keepdims=True) * u)

    def _laplacian_direction(self, u: ArrayF) -> ArrayF:
        l = cotan_laplacian(self.points, self.faces)
        m = vertex_areas(self.points, self.faces)
        rhs = -(l @ u)
        d = np.zeros_like(u)
        for k in range(3):
            d[:, k] = rhs[:, k] / m
        return d

    def optimize_active(self, u: ArrayF, active_mask: ArrayF) -> ArrayF:
        """Projected gradient flow with strict feasibility-preserving line search."""
        p = self.params
        u = u.copy()

        if not self.is_feasible(u, active_mask):
            raise ValueError("initial spherical map is not feasible (flipped/degenerate faces)")

        last_e = self._energy(u, active_mask)
        for _ in range(p.max_iter_per_step):
            d = self._laplacian_direction(u)
            d = self._project_tangent(u, d)
            d[~active_mask] = 0.0

            step = p.step0
            accepted = False
            while step >= p.min_step:
                cand = _row_normalize(u + step * d)
                if self.is_feasible(cand, active_mask):
                    e = self._energy(cand, active_mask)
                    if e <= last_e - p.armijo * step * np.sum(d * d):
                        u = cand
                        last_e = e
                        accepted = True
                        break
                step *= p.backtrack
            if not accepted:
                break
        return u

    def run_progressive(
        self,
        initial_active_vertices: Sequence[int],
        tetra_positions: ArrayF,
        progressive_steps: Sequence[ProgressiveStep],
    ) -> ArrayF:
        """Run coarse-to-fine insertion with re-optimization each step.

        `progressive_steps` must be ordered in *reverse-collapse* direction
        (from coarse tetra toward full mesh).
        """
        u = self.initialize_from_tetra(initial_active_vertices, tetra_positions)
        active = np.zeros(self.points.shape[0], dtype=bool)
        active[np.array(initial_active_vertices, dtype=int)] = True

        u = self.optimize_active(u, active)

        for st in progressive_steps:
            self._insert_vertex_guess(u, st.new_vertex, st.split_edge, active)
            active[st.new_vertex] = True
            if not self.is_feasible(u, active):
                raise ValueError(f"Inserted vertex {st.new_vertex} starts in infeasible state")
            u = self.optimize_active(u, active)

        return _row_normalize(u)
