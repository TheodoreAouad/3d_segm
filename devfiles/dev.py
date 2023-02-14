# from devfiles.dev3 import banana

from argparse import ArgumentParser, Action
import deep_morpho.experiments.parser as parser


# is_set = set() #global set reference
# class IsStored(Action):
#     def __call__(self, parser, namespace, values, option_string=None):
#         is_set.add(self.dest) # save to global reference
#         setattr(namespace, self.dest + '_set', True) # or you may inject directly to namespace
#         setattr(namespace, self.dest, values) # implementation of store_action
        # You cannot inject directly to self.dest until you have a custom class


# prs = parser.MultiParser()
prs = parser.Parser()
# prs["root"] = "banana"
# prs["preprocessing"] = "azea"

prs["dataset"] = "cifar10dataset"
prs["model"] = "BiMoNN"

# prs.parse_args()
# prs.parse_args(["-m", "BiMoNN", "-d", "cifar10dataset"])

# # print(banana)

# prs = ArgumentParser()
# # prs.add_argument("-m", "--model", nargs="+", help="Model")
# prs.add_argument("--kernel_size", nargs="+", action=IsStored)

# args = prs.parse_args(["--data", "banane"])
# args = prs.parse_args(["--n_inputs", "100"])
# prs["model"] = ["BiMoNN"]
# prs["dataset"] = ["cifar10dataset"]

args = prs.parse_args(["--kernel_size", "3"])

print(prs)
print(args)
# print(prs.multi_args[0]["kernel_size.net"])



# parser = ParserParent()
# print(parser.child1.parse_args())


pass
