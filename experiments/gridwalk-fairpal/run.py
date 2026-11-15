"""gridwalk-fairpal: gridwalk2's PAL arm with a *neutral* one-shot code demo (a shop-inventory task) instead of the
robot simulator, so the model must write its own direction logic. Everything else (tasks, seeds, prompts, parser)
is gridwalk2/run.py, imported and reused. Usage: same args as gridwalk2/run.py."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gridwalk2"))
import run as gw
gw.EX_Q = ("A shop starts with 5 apples and 3 pears. It processes these events in order:\n"
           "1. sell 2 apples\n2. receive 4 pears\n3. sell 1 pear\n4. receive 3 apples\n5. sell 2 pears\n"
           "How many (apples, pears) does it have at the end?")
gw.EX_PAL = ("```python\n"
             "stock = {'apples': 5, 'pears': 3}\n"
             "events = [('sell', 'apples', 2), ('receive', 'pears', 4), ('sell', 'pears', 1),\n"
             "          ('receive', 'apples', 3), ('sell', 'pears', 2)]\n"
             "for kind, item, n in events:\n"
             "    if kind == 'sell':\n"
             "        stock[item] -= n\n"
             "    elif kind == 'receive':\n"
             "        stock[item] += n\n"
             "print((stock['apples'], stock['pears']))\n```")
if __name__ == "__main__":
    gw.main()
