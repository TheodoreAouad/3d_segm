import re

args_path = "deep_morpho/results/results_tensorboards/Bimonn_exp_80/sandbox/positive_weights/0_/5/mnistclassifchannel/BimonnDenseNotBinary/version_0/args.yaml"

with open(args_path, "r") as f:
    input_text = f.read()

# pattern = r'threshold_mode\.net:(.*?)(?=^\s*\w+\.)'
# pattern = r'threshold_mode\.net:\s*\n\s*activation:\s*(.*?)\s*\n'
# pattern = r'threshold_mode\.net:\s*\n\s*weight:\s*(.*?)\s*\n'
# pattern = r'loss_coefs:.*?loss_data:(\s*(.*?)\s*(?:\n|$))'
pattern = r'loss_coefs:.*?loss_regu:(\s*(.*?)\s*(?:\n|$))'
matches = re.findall(pattern, input_text, re.DOTALL)

print(matches)
# if matches:
#     output = matches.group(1).strip()
#     print(output)
