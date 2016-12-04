from imageDirInfo import ImageDirInfo
import matplotlib.pyplot as plt
import sys

args = sys.argv
idi = ImageDirInfo(args[1])
idi.get_brightness_info()
plt.show()
