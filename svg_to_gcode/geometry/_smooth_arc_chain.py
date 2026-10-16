import math

from svg_to_gcode.geometry import Chain
from svg_to_gcode.geometry import Curve, Line, Vector
from svg_to_gcode.geometry import CircularArc, CubicBazier, QuadraticBezier, EllipticalArc
from svg_to_gcode import TOLERANCES

# Lengths below this are treated as zero (a vector with no direction, a degenerate chord).
_EPSILON = 1e-12

# How many times a bezier may be cut in half. 2^16 pieces is far more than any real curve needs, so in practice this
# only stops the recursion on pathological input.
_MAX_DEPTH = 16

# Number of samples used to measure the deviation between a curve and its arcs.
# _CURVE_SAMPLES: points taken along the bezier, each one measured against the arcs (bezier -> arcs).
# _ARC_SAMPLES: interior points taken along each arc, each one measured against the bezier (arcs -> bezier).
_CURVE_SAMPLES = 8
_ARC_SAMPLES = 3

# Nearest point on a bezier: first scan _NEAREST_SAMPLES evenly spaced parameters, then run _NEAREST_ITERATIONS steps of
# ternary search around the best one. Each step shrinks the interval to 2/3, so the interval ends up ~1e-3 of its size.
_NEAREST_SAMPLES = 16
_NEAREST_ITERATIONS = 14

# The deviation is only sampled, so the true maximum may slightly exceed the sampled one. Keep a margin below error_cap.
_SAFETY_MARGIN = 0.9

# An ellipse is split into cubic bezier pieces of at most this angle. The error of such a piece is below 1e-7 * radius.
_MAX_ELLIPSE_PIECE = math.pi / 8


class SmoothArcChain(Chain):
    """
    The SmoothArcChain class inherits from the abstract Chain class. It represents a continuous series of circular arcs.
    Straight line-segments are used where the curvature is too small to be represented by an arc.

    SmoothArcChains can be instantiated conventionally or through the static method arc_approximation(), which
    approximates any Curve with arcs and line-segments contained in a new SmoothArcChain instance.
    """

    def __repr__(self):
        return f"SmoothArcs({[curve.__repr__() for curve in self._curves]})"

    def append(self, curve):
        # Unlike the base class, nothing is changed here: the arcs were computed to match exactly, and moving a point
        # would break the tangency between neighbouring arcs.
        if self._curves:
            last = self._curves[-1]

            # Assert continuity
            if abs(last.end - curve.start) > TOLERANCES['input']:
                raise ValueError(f"The end of the last curve is different from the start of the new curve, "
                                 f"|{last.end} - {curve.start}| >= {TOLERANCES['input']}")

        self._curves.append(curve)

    @staticmethod
    def arc_approximation(shape, error_cap=None, max_depth=_MAX_DEPTH) -> "SmoothArcChain":
        """
        Approximate any shape with circular arcs.

        Each cubic bezier is replaced by a biarc: two arcs which are tangent to the curve at its ends and tangent to each
        other at the point where they meet. If the deviation exceeds error_cap, the bezier is split in half and each half
        is approximated separately.

        :param shape: The shape to be approximated.
        :param error_cap: the maximum acceptable distance between the shape and the arcs.
        :param max_depth: the maximum number of times a bezier may be split in half. Past it, a straight line is used.
        :return: A SmoothArcChain which approximates the given shape.
        """

        error_cap = TOLERANCES['approximation'] if error_cap is None else error_cap

        if error_cap <= 0:
            raise ValueError(f"This algorithm is approximate. error_cap must be a non-zero positive float. Not {error_cap}")

        # The result. Arcs and lines are appended to it in order, from the start of the shape to its end.
        arcs = SmoothArcChain()

        # Lines and arcs are already what the chain is made of. Nothing to approximate.
        if isinstance(shape, (Line, CircularArc)):
            arcs.append(shape)
            return arcs

        # Every other curve is reduced to cubic beziers, which are fitted one by one.
        for bezier in _to_cubic_beziers(shape):
            _fit(bezier, error_cap, max_depth, arcs)

        return arcs


def _to_cubic_beziers(shape):
    """Convert a shape to a list of cubic beziers that trace the same curve (exactly or to within ~1e-7)."""
    if isinstance(shape, CubicBazier):
        return [shape]

    if isinstance(shape, QuadraticBezier):
        # Degree elevation: the cubic bezier below traces exactly the same curve as the quadratic one.
        control1 = shape.start + 2 / 3 * (shape.control - shape.start)
        control2 = shape.end + 2 / 3 * (shape.control - shape.end)
        return [CubicBazier(shape.start, shape.end, control1, control2)]

    if isinstance(shape, EllipticalArc):
        return _ellipse_to_cubic_beziers(shape)

    raise TypeError(f"Cannot approximate {type(shape)} with arcs.")


def _ellipse_to_cubic_beziers(ellipse: EllipticalArc):
    """Approximate an elliptical arc with cubic beziers, each covering at most _MAX_ELLIPSE_PIECE radians."""

    # Split the sweep into equal pieces, so that every piece is small enough for the bezier approximation to be exact.
    pieces = max(1, math.ceil(abs(ellipse.sweep_angle) / _MAX_ELLIPSE_PIECE))
    piece_angle = ellipse.sweep_angle / pieces

    # The standard cubic bezier approximation of an arc: control points lie on the tangents at a distance of
    # 4/3 * tan(angle / 4) times the derivative with respect to the angle. It also holds for a transformed ellipse
    # because affine transformations map beziers to beziers.
    handle = 4 / 3 * math.tan(piece_angle / 4)

    # The derivative is estimated with a central difference, which includes the transformation of the ellipse.
    step = 1e-5

    # Piece i runs from angles[i] to angles[i + 1]. Neighbouring pieces share their end points and tangents.
    angles = [ellipse.start_angle + piece_angle * i for i in range(pieces + 1)]
    points = [ellipse.angle_to_point(angle) for angle in angles]
    tangents = [(ellipse.angle_to_point(angle + step) - ellipse.angle_to_point(angle - step)) / (2 * step)
                for angle in angles]

    # Control point 1 leaves the start along its tangent, control point 2 comes into the end along its tangent.
    return [CubicBazier(points[i], points[i + 1], points[i] + handle * tangents[i],
                        points[i + 1] - handle * tangents[i + 1]) for i in range(pieces)]


def _fit(bezier: CubicBazier, error_cap, depth, arcs: SmoothArcChain):
    """Append arcs approximating the bezier to the chain."""

    # A bezier smaller than the operational tolerance is a point. It can't be drawn.
    control_points = (bezier.control1, bezier.control2, bezier.end)
    if all(abs(point - bezier.start) < TOLERANCES['operation'] for point in control_points):
        return

    # Try to represent the whole bezier with a single biarc.
    candidate = _biarc(bezier)

    # The deviation is only sampled, so be stricter than error_cap. See _SAFETY_MARGIN.
    limit = error_cap * _SAFETY_MARGIN

    if candidate is not None and _deviation(bezier, candidate, limit) <= limit:
        for curve in candidate:
            arcs.append(curve)
        return

    # Out of splits. The pieces are tiny by now, so a straight line is accurate enough. This also guarantees termination.
    if depth == 0:
        arcs.append(Line(bezier.start, bezier.end))
        return

    # The biarc is not accurate enough (or doesn't exist). Each half of the bezier is flatter and simpler to fit.
    first, second = _split(bezier)

    # Fit both halves in order. Every call appends to the same chain, so the arcs stay in the order of the curve.
    _fit(first, error_cap, depth - 1, arcs)
    _fit(second, error_cap, depth - 1, arcs)


def _split(bezier: CubicBazier):
    """Split a bezier at t=0.5 with De Casteljau's algorithm."""
    p0, p1, p2, p3 = bezier.start, bezier.control1, bezier.control2, bezier.end

    # Repeated midpoints. The two new beziers trace exactly the same curve as the original.
    p01, p12, p23 = (p0 + p1) / 2, (p1 + p2) / 2, (p2 + p3) / 2
    p012, p123 = (p01 + p12) / 2, (p12 + p23) / 2
    middle = (p012 + p123) / 2

    return CubicBazier(p0, middle, p01, p012), CubicBazier(middle, p3, p123, p23)


def _unit(vector: Vector):
    """The vector scaled to length 1, or None for a vector too short to have a direction."""
    length = abs(vector)
    return None if length < _EPSILON else vector / length


def _end_tangents(bezier: CubicBazier):
    """Unit tangents at both ends. A control point which coincides with its end is skipped."""
    p0, p1, p2, p3 = bezier.start, bezier.control1, bezier.control2, bezier.end

    # The tangent at the start points to the first control point. If it coincides with the start, the next one defines
    # the tangent, and so on. The same logic applies to the end.
    start = next((t for t in map(_unit, (p1 - p0, p2 - p0, p3 - p0)) if t is not None), None)
    end = next((t for t in map(_unit, (p3 - p2, p3 - p1, p3 - p0)) if t is not None), None)

    return start, end


def _biarc(bezier: CubicBazier):
    """
    Build a biarc from the end points and end tangents of a bezier. Return None if there's no suitable biarc.

    Let T0 and T1 be the unit tangents at the start P0 and the end P1. Move along the tangents by d, to A = P0 + d*T0
    and B = P1 - d*T1, and pick d such that |AB| = 2d. The join point J is the middle of AB. Since |AJ| = |AP0| = d,
    the first arc is tangent to P0A at P0 and to AB at J. Since |BJ| = |BP1| = d, the second arc is tangent to AB at J
    and to P1B at P1. The two arcs are therefore tangent to each other at J.
    |P1 - P0 - d(T0 + T1)| = 2d is a quadratic in d with exactly one positive root.
    """
    start, end = bezier.start, bezier.end
    t0, t1 = _end_tangents(bezier)
    chord = end - start

    if t0 is None or t1 is None or abs(chord) < _EPSILON:
        return None

    # Solve |chord - d * (t0 + t1)| = 2d for d. See the docstring for the meaning.
    # Squaring both sides and expanding gives: 2(1 - t0.t1) d^2 + 2 (chord.(t0 + t1)) d - chord.chord = 0, which is
    # quadratic * d^2 + 2 * projection * d - chord.chord = 0 with the names below (the * operator is the dot product).
    tangent_sum = t0 + t1
    projection = chord * tangent_sum
    quadratic = 2 * (1 - t0 * t1)

    # The square root in the quadratic formula (the discriminant is already divided by 4 because of the factor 2 in
    # front of `projection`). max() protects from a tiny negative value caused by rounding.
    discriminant = math.sqrt(max(0, projection ** 2 + quadratic * (chord * chord)))

    # Both forms are the positive root. The first avoids dividing by zero when the tangents are parallel, the second
    # avoids cancellation when the chord points against the tangents.
    if projection >= 0:
        d = (chord * chord) / (projection + discriminant)
    elif quadratic > _EPSILON:
        d = (discriminant - projection) / quadratic
    else:
        # The tangents are parallel and point away from the chord. No biarc exists.
        return None

    # A and B are the corners of the tangent construction, the arcs meet at the middle of AB.
    a = start + d * t0
    b = end - d * t1
    join = (a + b) / 2
    join_tangent = _unit(b - a)

    if join_tangent is None:
        return None

    # The second arc leaves the join point along AB.
    first = _tangent_arc(start, t0, join)
    second = _tangent_arc(join, join_tangent, end)

    if first is None or second is None:
        return None

    return [first, second]


def _tangent_arc(start: Vector, tangent: Vector, end: Vector):
    """
    The circular arc from start to end which leaves start in the direction of the tangent. A line is returned if the
    arc is flatter than the operational tolerance. None is returned if the arc would be longer than a half circle.
    """
    chord = end - start
    chord_length = abs(chord)

    if chord_length < _EPSILON:
        return None

    # The normal points to the left of the direction of travel. The center of a counter-clockwise arc lies on it.
    normal = Vector(-tangent.y, tangent.x)

    # The chord in the coordinates of the tangent: `along` it and `across` it (positive to the left, negative to the
    # right). The sign of `across` is therefore the direction in which the arc turns.
    along = chord * tangent
    across = chord * normal

    # The angle between the tangent and the chord is half of the angle swept by the arc (tangent-chord angle theorem).
    half_sweep = math.atan2(across, along)

    if abs(half_sweep) > math.pi / 2:
        return None

    # The sagitta is the distance from the middle of the chord to the middle of the arc.
    sagitta = chord_length / 2 * math.tan(abs(half_sweep) / 2)
    if sagitta < TOLERANCES['operation']:
        return Line(start, end)

    # The center lies on the normal through the start, at the same distance from the start and from the end. Taking
    # center = start + r * normal in |center - end|^2 = r^2 gives r = |chord|^2 / (2 * across).
    # The center is at the signed radius along the normal. The center of a counter-clockwise arc is to the left of the
    # tangent (positive radius), the center of a clockwise arc is to the right (negative radius).
    signed_radius = (chord * chord) / (2 * across)
    center = start + signed_radius * normal

    return CircularArc(start, end, center, clockwise=signed_radius < 0)


def _distance_to_segment(line: Line, point: Vector):
    """The shortest distance from a point to a line-segment (not to the infinite line)."""
    direction = line.end - line.start
    length_squared = direction * direction

    # The segment is a point
    if length_squared == 0:
        return abs(point - line.start)

    # Project the point onto the line, then clamp the projection to the segment.
    t = min(1, max(0, ((point - line.start) * direction) / length_squared))
    return abs(point - (line.start + t * direction))


def _distance(curve, point: Vector):
    """The shortest distance from a point to an element of the chain, which is either a CircularArc or a Line."""
    if isinstance(curve, CircularArc):
        return curve.distance_to(point)

    # Anything else is a straight line
    return _distance_to_segment(curve, point)


def _distance_to_bezier(bezier: CubicBazier, point: Vector):
    """The shortest distance from a point to the bezier: coarse sampling followed by ternary search."""
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = bezier.start, bezier.control1, bezier.control2, bezier.end
    px, py = point

    # Plain floats instead of Vectors. This function runs a lot of times and Vectors are slow.
    def squared_distance(t):
        u = 1 - t
        a, b, c, d = u * u * u, 3 * u * u * t, 3 * u * t * t, t * t * t
        return (a * x0 + b * x1 + c * x2 + d * x3 - px) ** 2 + (a * y0 + b * y1 + c * y2 + d * y3 - py) ** 2

    # Coarse scan: a bezier can come close to the point in several places, so the minimum of the samples picks the
    # right neighbourhood before the ternary search narrows it down.
    ts = [i / _NEAREST_SAMPLES for i in range(_NEAREST_SAMPLES + 1)]
    distances = [squared_distance(t) for t in ts]
    nearest = distances.index(min(distances))

    # The nearest point lies near the nearest sample. Narrow it down with a ternary search.
    low = ts[max(0, nearest - 1)]
    high = ts[min(_NEAREST_SAMPLES, nearest + 1)]

    for _ in range(_NEAREST_ITERATIONS):
        third = (high - low) / 3
        if squared_distance(low + third) < squared_distance(high - third):
            high -= third
        else:
            low += third

    # The best of the scan and the search result. The scan is kept in case the search settled on a worse point.
    return math.sqrt(min(min(distances), squared_distance((low + high) / 2)))


def _deviation(bezier: CubicBazier, curves, error_cap):
    """
    The largest distance between the bezier and the arcs, measured in both directions: from points on the bezier to the
    arcs, and from points on the arcs to the bezier. The measurement stops as soon as error_cap is exceeded.
    """
    deviation = 0

    # Why both directions: with only bezier -> arcs, an arc could bulge away from the curve between the samples
    # without being noticed. With only arcs -> bezier, a part of the curve could be left uncovered.

    # From the bezier to the arcs. A point of the bezier is as far from the arcs as from the nearest one of them.
    for i in range(_CURVE_SAMPLES + 1):
        point = bezier.point(i / _CURVE_SAMPLES)
        deviation = max(deviation, min(_distance(curve, point) for curve in curves))

        if deviation > error_cap:
            return deviation

    # From the arcs to the bezier. Lines are skipped: they only replace arcs flatter than the operational tolerance.
    for curve in curves:
        if isinstance(curve, Line):
            continue

        for i in range(_ARC_SAMPLES):
            point = curve.point((i + 1) / (_ARC_SAMPLES + 1))
            deviation = max(deviation, _distance_to_bezier(bezier, point))

            if deviation > error_cap:
                return deviation

    return deviation
