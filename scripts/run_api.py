from dataclasses import replace

import uvicorn

from waveslide.api import create_app
from waveslide.config import config_from_env


# Edit these values directly instead of passing CLI arguments.
HOST = "127.0.0.1"
PORT = 8000
SIMULATE = False


def main() -> None:
    config = replace(
        config_from_env(),
        simulate=SIMULATE,
        control_presentation=False,
    )
    uvicorn.run(create_app(config), host=HOST, port=PORT)


if __name__ == "__main__":
    main()
