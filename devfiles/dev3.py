from importlib import import_module

# from devfiles.dev import prs
modl = import_module("devfiles.dev")
prs = modl.prs

print(prs)
