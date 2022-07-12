import neptune.new as neptune


class NeptuneLogger():
    """
    Neptune logger. Use it the same way you would use neptune run

    e.g
    n = NeptuneLogger()
    n['lr'] = 0.1
    n['loss'].log(0.01)
    """

    def __init__(self, exp_name):
        self.api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2MmVjN2EzYS04Y2FmLTRkYjItOTkyMi1mNmEwYWQzM2I3Y2UifQ=="
        self.project = f"rm360179/DistilHerBERT"
        self.run = neptune.init(project=self.project, api_token=self.api_token, name=exp_name)

    def __setitem__(self, key, val):
        self.run[key] = val

    def __getitem__(self, key):
        return self.run[key]

    def __del__(self):
        self.run.stop()

class WandbLogger():
    def __init__(self, wandb, exp_name, epochs):
        self.api_token = "07a2cd842a6d792d578f8e6c0978efeb8dcf7638"
        self.project = f"DistilHerBERT"
        wandb.login(key=self.api_token)
        wandb.init(
            # Set the project where this run will be logged
            project=self.project,
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=exp_name,
            # Track hyperparameters and run metadata
            config={
            "learning_rate": 0.02,
            "architecture": "BERT",
            "dataset": "cc100-pl",
            "epochs": epochs,
        })
