"""Wire all nodes to the bus and run.

    python -m so101_assist.main --config config/default.yaml

Each node runs in its own thread; the arm control loop gets priority.
Startup order: driver -> safety -> cameras -> UI -> perception ->
voice -> inputs -> state machine. Ctrl-C stops motion, then exits.
"""

def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
