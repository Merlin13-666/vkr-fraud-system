import uvicorn

from fraud_system.api.app import create_app
from fraud_system.api.settings import ApiSettings


def main():
    # Берём всё из env (включая FRAUD_API_API_KEY)
    settings = ApiSettings.from_env()
    app = create_app(settings)

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    main()