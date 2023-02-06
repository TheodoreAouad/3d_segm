# from devfiles.dev3 import banana

from argparse import ArgumentParser
import deep_morpho.experiments.parser as parser


prs = parser.Parser()
# prs["root"] = "banana"
# prs["preprocessing"] = "azea"

# prs["dataset"] = "cifar10dataset"
# prs["model"] = "BiMoNN"

# prs.parse_args()
# prs.parse_args(["-m", "BiMoNN", "-d", "cifar10dataset"])

# # print(banana)

prs = ArgumentParser()
# prs.add_argument("-m", "--model", help="Model")


def str_to_list(s):
    return s.replace(" ", "").replace("[", "").replace("]", "").split(",")


prs.add_argument("-d", "--dataset", type=str_to_list, help="Dataset")

# args = prs.parse_args(["--data", "banane"])
args = prs.parse_args()

print(args)



# parser = ParserParent()
# print(parser.child1.parse_args())


pass
