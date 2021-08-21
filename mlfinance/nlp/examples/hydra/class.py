import hydra


# hydra opens model_1.yaml
@hydra.main(config_path=".", config_name="model_1")
def hydra_entry_1(cfg):
    # it calls the function tests.example_model.Model and puts the
    model = hydra.utils.instantiate(cfg.call)
    print(model.name)
    print(model.lr)


@hydra.main(config_path=".", config_name="model_2")
def hydra_entry_2(cfg):
    model = hydra.utils.instantiate(cfg.call)
    print(model.name)
    print(model.lr)


if __name__ == "__main__":
    hydra_entry_1()
    hydra_entry_2()
