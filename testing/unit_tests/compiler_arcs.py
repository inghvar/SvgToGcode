"""Verify that Compiler.append_curves emits GRBL compatible G2/G3 code. Run with:
python3 -m unittest testing.unit_tests.compiler_arcs"""

import math
import re
import unittest

from svg_to_gcode.compiler import Compiler
from svg_to_gcode.compiler.interfaces import Gcode
from svg_to_gcode.geometry import CubicBazier, Line, Vector


def compile_curves(curves, **kwargs):
    """Compile the curves with arcs, one pass, and return the code as a string."""
    compiler = Compiler(Gcode, movement_speed=1000, cutting_speed=300, pass_depth=0, **kwargs)
    compiler.append_curves(curves)
    return compiler.compile(passes=1)


def motion_lines(code):
    """The lines of the code which move the tool: G0, G1, G2 and G3."""
    return [line for line in code.split('\n') if re.match(r'G[0-3]\b', line)]


class CompilerArcsTest(unittest.TestCase):

    def test_cubic_is_drawn_with_arcs(self):
        # The code contains G2/G3 and is much shorter than the G1-only code.
        code = compile_curves([CubicBazier(Vector(0, 0), Vector(0, 30), Vector(40, 30), Vector(40, 0))])

        arcs = [line for line in motion_lines(code) if line.startswith(('G2', 'G3'))]
        self.assertGreater(len(arcs), 0)
        # A cubic of this size needs far fewer than the ~80 line-segments of the old algorithm.
        self.assertLess(len(motion_lines(code)), 60)

    def test_arcs_satisfy_grbl_radius_check(self):
        code = compile_curves([CubicBazier(Vector(0, 0), Vector(0, 30), Vector(40, 30), Vector(40, 0))])

        # Read the code back like GRBL does. (x, y) is the current position, updated after every move.
        x = y = None
        for line in motion_lines(code):
            words = {key: float(value) for key, value in re.findall(r'([XYIJ])(-?[\d.]+)', line)}
            if line.startswith(('G2', 'G3')):
                # GRBL rejects an arc whose start and end radii differ by more than 0.005 mm.
                cx, cy = x + words['I'], y + words['J']
                self.assertAlmostEqual(math.hypot(x - cx, y - cy), math.hypot(words['X'] - cx, words['Y'] - cy),
                                       delta=0.005)
                self.assertNotEqual((words['X'], words['Y']), (x, y))  # would be read as a full circle
            x, y = words.get('X', x), words.get('Y', y)

    def test_closed_loop_has_no_full_circle_arc(self):
        # A closed loop must never be written as one arc with end == start, GRBL would read it as a full circle
        # with an undefined direction of travel.
        code = compile_curves([CubicBazier(Vector(0, 0), Vector(10, 10), Vector(-10, 10), Vector(0, 0))])

        x = y = None
        for line in motion_lines(code):
            words = {key: float(value) for key, value in re.findall(r'([XYIJ])(-?[\d.]+)', line)}
            if line.startswith(('G2', 'G3')):
                self.assertNotEqual((words['X'], words['Y']), (x, y))
            x, y = words.get('X', x), words.get('Y', y)

    def test_straight_line_stays_g1(self):
        # A straight line has no curvature, so it is drawn as it was: with G1.
        code = compile_curves([Line(Vector(0, 0), Vector(10, 5))])

        self.assertNotIn("G2", code)
        self.assertNotIn("G3", code)
        self.assertIn("G1 F300 X10.000000 Y5.000000;", code)

    def test_connected_curves_are_cut_in_one_go(self):
        # Two curves in a row, the second one starting at the end of the first.
        first = Line(Vector(0, 0), Vector(10, 0))
        second = Line(Vector(10, 0), Vector(10, 10))
        code = compile_curves([first, second])

        # The laser is turned on once: the second curve starts where the first one ended.
        self.assertEqual(code.count("M3 S"), 1)

    def test_laser_power_is_passed_to_the_interface(self):
        # The default is full power (S255), a given power is scaled by the interface.
        line = [Line(Vector(0, 0), Vector(10, 5))]
        self.assertIn("M3 S255;", compile_curves(line))
        self.assertIn("M3 S127.5;", compile_curves(line, laser_power=0.5))

    def test_laser_power_out_of_bounds(self):
        with self.assertRaises(ValueError):
            Compiler(Gcode, 1000, 300, 0, laser_power=1.5)


if __name__ == '__main__':
    unittest.main()
