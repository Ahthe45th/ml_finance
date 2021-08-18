import hydra
from omegaconf import DictConfig




@hydra.main(config_path="./conf", config_name="config")
def hydra_entry(cfg: DictConfig) -> None:
    print(cfg)
    pass


if __name__ == "__main__":
    hydra_entry()