import hydra
from omegaconf import DictConfig



def main(cfg):
    """
training, testing, and evaluating done here
    """
    if cfg.mode == 'train':
        pass
    elif cfg.mode == 'test':
        pass
    else:
        pass



@hydra.main(config_path="./conf", config_name="config")
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg)



if __name__ == "__main__":
    hydra_entry()