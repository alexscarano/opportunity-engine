import logging

log = logging.getLogger("api.worker")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    # ponytail: stub so the Cloud Run Job command is fixed now; B4 implements the run loop
    log.info("worker: nothing to run yet")


if __name__ == "__main__":
    main()
