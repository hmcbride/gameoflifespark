import numpy as np
import pandas as pd
import pyspark.pandas as ps
from pyspark.sql.functions import udf
# I started this as a script / set of instrucition at the command line
# I did get pretty bogged down in the very last steps so
# as I was determined to stick with the dataframes so
# never formalized the script into a regular file
ps.set_option('compute.ops_on_diff_frames', True)

n = 5
# Setting up the initial states of 1 and 0s, am padding
# the set with zeros this will help in applying the
# convolution process to count all the neighbours
seed = np.random.randint(2, size=(n, n)).astype(float)
padded_set = np.zeros((n + 2, n + 2), int).astype(float)
padded_set[1:6, 1:n + 1] = seed

padded_panda = pd.DataFrame(padded_set)
padded_pyspk = ps.DataFrame(padded_panda)

seed_panda = pd.DataFrame(seed)
seed_pyspk = ps.DataFrame(seed_panda)


# This essentially sets up the convolution to count the neighbours a
# node has,
mask = padded_pyspk.rolling(3).sum().transpose().rolling(3).sum().transpose()

# As this generates rows and columns of NaNs we remove and
# reset the row and column numbering
mask_interim = mask.drop([0, 1])
mask_final = mask_interim.tail(mask_interim.shape[0] - 2)
mask_final.reset_index(inplace=True, drop=True)
rename_val = dict(zip(mask_final.columns.to_list(), seed_pyspk.columns.to_list()))
maskx = mask_final.rename(rename_val, axis='columns')

# As there is double count we remove the orignal
# element to correct
nebr_count = maskx.subtract(seed_pyspk)

# renaming columns on both sets for convenience
enames = list(map(lambda x: "e_" + str(x), seed_pyspk.columns.to_list()))
nnames = list(map(lambda x: "n_" + str(x), seed_pyspk.columns.to_list()))

rename_seed = dict(zip(seed_pyspk.columns.to_list(), enames))
rename_des = dict(zip(seed_pyspk.columns.to_list(), nnames))

seedfin = seed_pyspk.rename(rename_seed, axis='columns')
desfin = nebr_count.rename(rename_des, axis='columns')

# We now have two set, the matrix of nodes that are alive or dead
# with the set of neighbours each of them have appended side by side
xc = ps.concat([seedfin, desfin], axis=1)

xcdf = xc.to_spark()


# the logic to determine the next state of the node
# based on the number of neighbours it has and
# wheter it is alive or dead ng for nextgen
def ng(c, n):
    if c == 1.0 and n < 2.0:
        return 0.0
    elif c == 1.0 and n >= 2.0 or n <= 3.0:
        return 0.0
    elif c == 0.0 and n > 3.0:
        return 1.0


# register the udf
nextgen = udf(ng, 'double')

# the first column of the next generation
nextgen_set = xcdf.select(nextgen('e_0', 'n_0'))
nextgen_set.show()
nextgen_set = nextgen_set.to_pandas_on_spark()

# generate a list fo tuples for the remaining columns
allcols = xcdf.columns
h = int(len(allcols)/2)
sh = allcols[h:]
fh = allcols[:h]
fh.pop(0)
sh.pop(0)
xlist = list(zip(fh, sh))

# iterate thru the rest of the data frame building
# up the dataframe ideally this should use a
# lambda func and imho this is not and ideal way
# to create the next generation but its what I
# could get to work plus there were issues with
# WindowPartiton which I was going to investigate
# if I had time
for i in xlist:
    t = xcdf.select(nextgen(i[0], i[1]))
    t = t.to_pandas_on_spark()
    nextgen_set = ps.concat([nextgen_set, t], axis=1)


# Give that our result set only contains two value
# partiontion the file on write will only ever
# create two directories , you usually use a field with higher
# cardinality , outside of that it default to the hdfs block size
# and will split files greater than that isze
nextgen_set.write.partitionBy("ng('e_1','n_1'").mode("overwrite") .csv("/tmp/finalset")



