"""Pipeline Orchestrator — wires all modules end-to-end.

Stub for Week 3. Will implement:
- email → download → extract → validate → store flow
- Configurable modes: backlog vs daily
- APScheduler cron for 6 PM IST daily runs
- Lambda handler entrypoint (Week 5)

Week 3 of the CSE Stock Market Data Pipeline.
"""
# TODO (Week 3): Implement pipeline orchestrator and scheduler


def main() -> None:
    """CLI entry point: ``cse-pipeline`` (defined in pyproject.toml)."""
    from src.config import settings

    settings.configure_logging()
    raise NotImplementedError(
        "Pipeline orchestrator not yet implemented. Coming in Week 3."
    )


if __name__ == "__main__":
    main()
