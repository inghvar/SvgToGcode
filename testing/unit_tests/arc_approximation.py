"""Verify that curves are approximated by circular arcs within the approximation tolerance. Run with:
python3 -m unittest testing.unit_tests.arc_approximation"""

import math
import unittest

from svg_to_gcode import TOLERANCES
from svg_to_gcode.geometry import Vector, Line, CircularArc, CubicBazier, QuadraticBezier, EllipticalArc
from svg_to_gcode.geometry import SmoothArcChain
from svg_to_gcode.geometry._smooth_arc_chain import _distance, _distance_to_segment


def sampled_distance(curves, shape, samples=250):
    """Largest distance between a shape and a chain of curves, measured in both directions."""
    shape_points = [shape.point(i / samples) for i in range(samples + 1)]
    polyline = [Line(a, b) for a, b in zip(shape_points, shape_points[1:])]

    to_chain = max(min(_distance(curve, p) for curve in curves) for p in shape_points)

    from_chain = 0
    for curve in curves:
        points = [curve.start, curve.end] if isinstance(curve, Line) else [curve.point(i / 10) for i in range(11)]
        for p in points:
            from_chain = max(from_chain, min(_distance_to_segment(segment, p) for segment in polyline))

    return max(to_chain, from_chain)


def tangent_at_end(curve):
    """Unit vector in the direction of travel at the end of a Line or a CircularArc."""
    if isinstance(curve, Line):
        direction = curve.end - curve.start
        return direction / abs(direction)

    return curve.tangent(1)


def tangent_at_start(curve):
    """Unit vector in the direction of travel at the start of a Line or a CircularArc."""
    if isinstance(curve, Line):
        direction = curve.end - curve.start
        return direction / abs(direction)

    return curve.tangent(0)


class ArcApproximationTest(unittest.TestCase):

    def assert_valid_chain(self, shape, curves):
        """The checks every approximation must pass: connected, on valid arcs and within the tolerance."""
        self.assertGreater(len(curves), 0)

        # The chain starts and ends where the shape does.
        self.assertLess(abs(curves[0].start - shape.start), TOLERANCES["operation"])
        self.assertLess(abs(curves[-1].end - shape.end), TOLERANCES["operation"])

        # Pieces shorter than the operational tolerance are dropped, which leaves a gap of at most that size.
        for previous, current in zip(curves, curves[1:]):
            self.assertLess(abs(previous.end - current.start), TOLERANCES["operation"])

        # Every arc is a real arc: the end is on the same circle as the start, and a G2/G3 can't exceed a half circle
        # in this algorithm.
        for curve in curves:
            if isinstance(curve, CircularArc):
                self.assertLess(abs(curve.radius - abs(curve.end - curve.center)), 1e-6)
                self.assertLessEqual(abs(curve.sweep_angle), math.pi + 1e-9)

        # The arcs stay within the approximation tolerance from the shape, measured independently of the algorithm.
        self.assertLessEqual(sampled_distance(curves, shape), TOLERANCES["approximation"])

    def approximate(self, shape):
        """Approximate the shape with arcs, check the result and return the list of arcs and lines."""
        curves = list(SmoothArcChain.arc_approximation(shape))
        self.assert_valid_chain(shape, curves)
        return curves

    def test_circular_arc_direction(self):
        # The same end points and center describe two arcs: the short one (quarter circle) and the long one (three
        # quarters). The direction flag chooses between them.
        counter_clockwise = CircularArc(Vector(1, 0), Vector(0, 1), Vector(0, 0), clockwise=False)
        clockwise = CircularArc(Vector(1, 0), Vector(0, 1), Vector(0, 0), clockwise=True)

        self.assertAlmostEqual(counter_clockwise.sweep_angle, math.pi / 2)
        self.assertAlmostEqual(clockwise.sweep_angle, -3 * math.pi / 2)
        self.assertLess(abs(counter_clockwise.point(0.5) - Vector(math.sqrt(0.5), math.sqrt(0.5))), 1e-9)
        self.assertLess(abs(clockwise.point(0.5) - Vector(-math.sqrt(0.5), -math.sqrt(0.5))), 1e-9)

        # The center is at (0, 0), which is (-1, 0) from the start (1, 0). These are the I and J of G2/G3.
        self.assertLess(abs(counter_clockwise.center_offset() - Vector(-1, 0)), 1e-9)

    def test_circular_arc_distance(self):
        arc = CircularArc(Vector(1, 0), Vector(0, 1), Vector(0, 0), clockwise=False)

        # Outside the sweep of the arc: the nearest point is on the circle. Then below the arc: the nearest point is the
        # start. Then the center: every point of the circle is at the radius.
        self.assertAlmostEqual(arc.distance_to(Vector(2, 2)), math.sqrt(8) - 1)
        self.assertAlmostEqual(arc.distance_to(Vector(2, -1)), math.sqrt(2))
        self.assertAlmostEqual(arc.distance_to(Vector(0, 0)), 1)

    def test_random_cubic_beziers(self):
        import random

        # A fixed seed keeps the test reproducible.
        random.seed(7)

        for _ in range(12):
            points = [Vector(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(4)]
            curves = self.approximate(CubicBazier(*points))

            for previous, current in zip(curves, curves[1:]):
                if isinstance(previous, CircularArc) and isinstance(current, CircularArc):
                    # Smooth wherever the cubic is smooth. Splitting a bezier keeps its tangent.
                    self.assertGreater(tangent_at_end(previous) * tangent_at_start(current), 0.99)

    def test_s_curve(self):
        # An S shape turns both ways, so both G2 and G3 must be used.
        curves = self.approximate(CubicBazier(Vector(0, 0), Vector(10, 10), Vector(20, 0), Vector(0, 10)))
        self.assertTrue(any(c.clockwise for c in curves if isinstance(c, CircularArc)))
        self.assertTrue(any(not c.clockwise for c in curves if isinstance(c, CircularArc)))

    def test_straight_bezier_becomes_line(self):
        # All control points are on a line: there is no curvature to approximate with an arc.
        curves = self.approximate(CubicBazier(Vector(0, 0), Vector(10, 0), Vector(5, 0), Vector(20, 0)))
        self.assertTrue(all(isinstance(c, Line) for c in curves))

    def test_closed_loop(self):
        # The end coincides with the start (loops, and a first control point equal to the start).
        self.approximate(CubicBazier(Vector(0, 0), Vector(0, 0), Vector(30, 40), Vector(-30, 40)))
        self.approximate(CubicBazier(Vector(0, 0), Vector(40, 30), Vector(40, -30), Vector(0, 0)))

    def test_cusp(self):
        # A sharp corner: the tangent flips direction, so no single arc fits and the bezier has to be split.
        self.approximate(CubicBazier(Vector(0, 0), Vector(10, 10), Vector(-10, 10), Vector(0, 0.01)))

    def test_coincident_control_points(self):
        # A control point equal to its end point has no direction, the next control point must be used instead.
        self.approximate(CubicBazier(Vector(0, 0), Vector(0, 0), Vector(10, 5), Vector(10, 5)))
        self.approximate(CubicBazier(Vector(0, 0), Vector(20, 0), Vector(20, 0), Vector(20, 20)))

    def test_point_yields_empty_chain(self):
        # A curve that is a single point can't be drawn: the chain is empty, and the compiler skips empty chains.
        curves = list(SmoothArcChain.arc_approximation(CubicBazier(Vector(1, 1), Vector(1, 1), Vector(1, 1), Vector(1, 1))))
        self.assertEqual(curves, [])

    def test_quadratic_bezier(self):
        # A quadratic bezier is converted to a cubic one first.
        self.approximate(QuadraticBezier(Vector(0, 0), Vector(30, 0), Vector(15, 25)))

    def test_circle_needs_few_arcs(self):
        # A circle is already made of arcs: the result consists of arcs only, all with the radius of the circle.
        circle = EllipticalArc(Vector(50, 50), Vector(30, 30), 0, 0, 2 * math.pi, None)
        curves = self.approximate(circle)

        arcs = [c for c in curves if isinstance(c, CircularArc)]
        self.assertEqual(len(arcs), len(curves))
        self.assertTrue(all(abs(a.radius - 30) < TOLERANCES["approximation"] for a in arcs))

    def test_ellipse(self):
        # A rotated ellipse (30 degrees) with a sweep of more than a half turn.
        ellipse = EllipticalArc(Vector(10, 20), Vector(40, 15), math.radians(30), 0.3, -4.0, None)
        self.approximate(ellipse)

    def test_existing_line_and_arc_pass_through(self):
        # A Line is already a chain element, so it is returned as is.
        line = Line(Vector(0, 0), Vector(5, 5))
        self.assertEqual(list(SmoothArcChain.arc_approximation(line)), [line])


if __name__ == "__main__":
    unittest.main()
