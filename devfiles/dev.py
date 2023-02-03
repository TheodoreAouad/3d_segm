# from devfiles.dev3 import banana

# from argparse import ArgumentParser
import deep_morpho.experiments.parser as parser


prs = parser.Parser()
# prs["root"] = "banana"
# prs["preprocessing"] = "azea"

prs["dataset"] = "cifar10dataset"
prs["model"] = "BiMoNN"

prs.parse_args()
# prs.parse_args(["-m", "BiMoNN", "-d", "cifar10dataset"])

# # print(banana)
print(prs)



# parser = ParserParent()
# print(parser.child1.parse_args())


pass
