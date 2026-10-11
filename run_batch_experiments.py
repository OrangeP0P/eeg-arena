"""User-facing batch entrypoint.

This script intentionally keeps business logic in `experiments.batch.scheduler`,
while exposing a readable top-level flow for first-time users.
"""

from experiments.batch.scheduler import Scheduler, parse_args


def main(argv=None):
    """Parse CLI args and run the batch scheduler."""
    args = parse_args(argv)
    scheduler = Scheduler(args)
    scheduler.run()


if __name__ == "__main__":
    main()
