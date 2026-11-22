"""gridwalk-analogpal: gridwalk2's PAL arm with an *analogous* one-shot code demo: a 4-colour spinner (cyclic state,
clockwise/counter-clockwise turns with % 4, an action that depends on the current state, instructions transcribed into
a list). It shows the program structure but contains no grid, coordinates or direction vectors, so the model must map
turns and forward moves to x/y itself. Everything else (tasks, seeds, prompts, parser) is gridwalk2/run.py, reused."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gridwalk2"))
import run as gw
gw.EX_Q = ("A spinner has four colours in clockwise order: red, green, blue, yellow. It starts pointing at red, and "
           "every colour starts with 0 points. It follows these instructions in order:\n"
           "1. add 2 points\n2. spin clockwise\n3. add 3 points\n4. spin halfway round\n5. add 1 point\n"
           "6. spin counter-clockwise\n7. add 4 points\n"
           "What are the final points (red, green, blue, yellow)?")
gw.EX_PAL = ("```python\n"
             "colours = ['red', 'green', 'blue', 'yellow']  # clockwise = +1, counter-clockwise = -1, halfway = +2\n"
             "points = {c: 0 for c in colours}\n"
             "c = 0\n"
             "steps = [('add', 2), ('cw', 0), ('add', 3), ('half', 0), ('add', 1), ('ccw', 0), ('add', 4)]\n"
             "for act, n in steps:\n"
             "    if act == 'add':\n"
             "        points[colours[c]] += n\n"
             "    elif act == 'cw':\n"
             "        c = (c + 1) % 4\n"
             "    elif act == 'ccw':\n"
             "        c = (c - 1) % 4\n"
             "    elif act == 'half':\n"
             "        c = (c + 2) % 4\n"
             "print(tuple(points[k] for k in colours))\n```")
if __name__ == "__main__":
    gw.main()
