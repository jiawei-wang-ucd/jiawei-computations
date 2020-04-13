import cpuinfo
import pandas as pd
data = pd.DataFrame.from_dict({ key: str(value) for key, value in
cpuinfo.get_cpu_info().iteritems()}, orient='index')
data.to_csv("./system_info/cpuinfo.csv", index=True, header=False)
