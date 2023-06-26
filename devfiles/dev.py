import pydicom
from pydicom import tag

file = "/hdd/aouadt/projets/data challenge sfr 2023/data/IGR/NLASSAU_01_BENIN"

dcm = pydicom.read_file(file)

dcm[tag.Tag(0x7FE0, 0x0010)]
