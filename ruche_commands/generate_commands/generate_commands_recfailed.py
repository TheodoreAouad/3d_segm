TEMPLATE_PATH = "ruche_commands/templates/recompute_failed_morpho.sh"

with open(TEMPLATE_PATH, "r") as f:
    template = f.read()


for number in range(8):
    with open(f"ruche_commands/recompute_failed/args_{number}.sh", "w") as f:
            print(template.replace("{NUMBER}", f"{number}"), file=f)
