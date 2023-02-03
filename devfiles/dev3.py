from argparse import ArgumentParser


class ParserParent(ArgumentParser):
    def __init__(self):
        super().__init__()
        self.parent1 = ArgumentParser(add_help=False)
        self.parent1.add_argument("--parent1", default="parent1")

        self.parent2 = ArgumentParser(add_help=False)
        self.parent2.add_argument("--parent2", default="parent2")

        self.child1 = ArgumentParser(parents=[self.parent1], add_help=False)
        self.child1.add_argument("--child1", default="child1")

        self.child2 = ArgumentParser(parents=[self.parent2], add_help=False)
        self.child2.add_argument("--child1", default="child2")

        self.grandchild = ArgumentParser(parents=[self.child1, self.child2])


parser = ParserParent()
