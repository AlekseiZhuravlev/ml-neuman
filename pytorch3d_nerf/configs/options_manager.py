from lightning.pytorch.cli import LightningCLI

class CustomCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--notification_email", default="will@email.com")

