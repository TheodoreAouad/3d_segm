import pydicom
from pydicom import tag

file = "/hdd/aouadt/projets/data challenge sfr 2023/data/IGR/test/TAP.Seq4.Ser5.Img1.dcm"

dcm = pydicom.read_file(file)

dcm[tag.Tag(0x7FE0, 0x0010)]
