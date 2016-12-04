from imageDirInfo import ImageDirInfo
import matplotlib.pyplot as plt
import sys

args = sys.argv
idi = ImageDirInfo(sys.argv[1])
x, y = idi.get_all_sizes()
subPlot1 = idi.plot(x)
idi.plot(y)
subPlot1.legend(['Widths', 'Heights'], loc='upper left')
plt.show()
