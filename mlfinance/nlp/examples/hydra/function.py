import hydra


@hydra.main(config_path=".", config_name="function")
def hydra_entry(cfg):
    hydra.utils.call(cfg.call)


if __name__ == "__main__":
    hydra_entry()
