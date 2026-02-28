"""Minimal run entry for the slimmed TimeLens branch.

This branch intentionally keeps only core python modules:
- models/
- dataset/
- losses/
- run_network.py
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description='Minimal TimeLens branch entrypoint')
    parser.add_argument('--about', action='store_true', help='Print kept module layout and exit')
    args = parser.parse_args()

    if args.about:
        print('Kept modules: models/, dataset/, losses/, run_network.py')
        return

    print('This slim branch only keeps core module files. Use --about for details.')


if __name__ == '__main__':
    main()
