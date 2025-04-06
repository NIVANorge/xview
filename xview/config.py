import pydantic_settings


class Settings(pydantic_settings.BaseSettings):
    server_url: str = "http://localhost:8000"
    bokeh_url: str = "http://localhost:5000"

    class Config:
        env_file = ".env"


SETTINGS = Settings()
