import hydra
from omegaconf import DictConfig



def main(cfg):
    print(cfg)



@hydra.main(config_path="./conf", config_name="config")
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg)



if __name__ == "__main__":
    hydra_entry()