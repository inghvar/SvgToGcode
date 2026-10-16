import math

from svg_to_gcode.geometry import Vector
from svg_to_gcode.geometry import Curve
from svg_to_gcode import TOLERANCES


class CircularArc(Curve):
    """
    The CircularArc class inherits from the abstract Curve class and describes a circular arc.

    The arc runs from start to end around center in the given direction. The direction is stored explicitly because the
    endpoints and the center alone describe two different arcs (G2 and G3 in gcode).
    """

    __slots__ = 'center', 'clockwise', 'radius', 'start_angle', 'sweep_angle', 'end_angle'

    def __init__(self, start: Vector, end: Vector, center: Vector, clockwise: bool):
        self.start = start
        self.end = end
        self.center = center
        self.clockwise = clockwise

        # The radius is measured at the start. sanity_check() verifies that the end is on the same circle.
        self.radius = abs(self.start - self.center)
        self.start_angle = self.point_to_angle(self.start)

        # The sweep is the angle travelled from the start to the end, so it is the difference of the two angles. The
        # modulo wraps it into [0, 2π), that is the sweep of a counter-clockwise arc.
        # Counter-clockwise sweep in [0, 2π). A clockwise arc covers the remaining part of the circle in the other direction.
        sweep = (self.point_to_angle(self.end) - self.start_angle) % (2 * math.pi)
        if clockwise and sweep > 0:
            sweep -= 2 * math.pi

        self.sweep_angle = sweep
        self.end_angle = self.start_angle + self.sweep_angle

    # start_angle, sweep_angle and end_angle are in radians. Angles are measured counter-clockwise from the positive x
    # axis, like in mathematics. sweep_angle is negative for a clockwise arc.

    def __repr__(self):
        return f"Arc(start: {self.start}, end: {self.end}, center: {self.center}, clockwise: {self.clockwise})"

    def length(self):
        # Arc length = angle (in radians) * radius
        return abs(self.sweep_angle) * self.radius

    def center_offset(self):
        """The vector from the start point to the center. These are the I and J words of G2/G3 in gcode."""
        return self.center - self.start

    def angle_to_point(self, rad):
        """The point of the circle at the given angle."""
        at_origin = self.radius * Vector(math.cos(rad), math.sin(rad))
        translated = at_origin + self.center
        return translated

    def point_to_angle(self, point: Vector):
        """The angle of the direction from the center to the point, in (-π, π]."""
        offset = point - self.center
        return math.atan2(offset.y, offset.x)

    def point(self, t):
        # t = 0 is the start of the arc and t = 1 is the end. The angle changes linearly with t.
        return self.angle_to_point(self.start_angle + t * self.sweep_angle)

    def derivative(self, t):
        """The derivative of point(t) with respect to t. Its direction is the direction of travel."""
        angle = self.start_angle + t * self.sweep_angle
        return self.sweep_angle * self.radius * Vector(-math.sin(angle), math.cos(angle))

    def tangent(self, t):
        """The unit vector pointing in the direction of travel at point(t)."""
        angle = self.start_angle + t * self.sweep_angle
        # The tangent of a counter-clockwise circle is the position vector rotated by 90 degrees, clockwise is opposite.
        direction = -1 if self.clockwise else 1
        return direction * Vector(-math.sin(angle), math.cos(angle))

    def distance_to(self, point: Vector):
        """The shortest distance from a point to the arc."""
        offset = point - self.center

        # The center itself has no angle. Every point of the circle is equally far.
        if abs(offset) > 0:
            # The angle travelled from the start to the point, in the direction of the arc.
            travelled = math.atan2(offset.y, offset.x) - self.start_angle
            if self.clockwise:
                travelled = -travelled

            # If the point is within the sweep of the arc, the nearest point is on the circle straight toward it.
            if travelled % (2 * math.pi) <= abs(self.sweep_angle):
                return abs(abs(offset) - self.radius)

        # Otherwise the nearest point is one of the ends.
        return min(abs(point - self.start), abs(point - self.end))

    def sanity_check(self):
        # The checks below reject arcs which can't be drawn or which have inconsistent data.
        # Assert that the Arc is not a point or a line
        try:
            assert abs(self.start - self.end) > TOLERANCES["input"]
        except AssertionError:
            raise ValueError(f"Arc is a point. The start and the end points are equivalent: "
                             f"|{self.start} - {self.end}| <= {TOLERANCES['input']}")

        try:
            assert abs(self.start - self.center) > TOLERANCES["input"]
        except AssertionError:
            raise ValueError(f"Arc is a line. The start and the center points are equivalent, "
                             f"|{self.start} - {self.center}| <= {TOLERANCES['input']}")

        try:
            assert abs(self.end - self.center) > TOLERANCES["input"]
        except AssertionError:
            raise ValueError(f"Arc is a line. The end and the center points are equivalent, "
                             f"|{self.end} - {self.center}| <= {TOLERANCES['input']}")

        # Assert that the center is equidistant from the start and the end
        try:
            assert abs(abs(self.start - self.center) - abs(self.end - self.center)) < TOLERANCES['input']
        except AssertionError:
            raise ValueError(f"Center is not equidistant to the start and end points within tolerance, "
                             f"|{abs(self.start - self.center)} - {abs(self.end - self.center)}| >= {TOLERANCES['input']}")
