import yaml


class Config:
    def __init__(self, file: str) -> None:
        with open(file, "r", encoding="utf-8") as f:
            self.config = yaml.load(f.read(), Loader=yaml.FullLoader)


if __name__ == "__main__":
    config = Config("config/config.yaml")
    print(config.config)
