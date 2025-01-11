# Simple example of how to use the torchhd library
import torch, torchhd

d = 10000  # number of dimensions

# create the hypervectors for each symbol
keys = torchhd.random(3, d)
country, capital, currency = keys

usa, mex = torchhd.random(2, d)  # United States and Mexico
wdc, mxc = torchhd.random(2, d)  # Washington D.C. and Mexico City
usd, mxn = torchhd.random(2, d)  # US Dollar and Mexican Peso

# create country representations
us_values = torch.stack([usa, wdc, usd])
us = torchhd.hash_table(keys, us_values)

mx_values = torch.stack([mex, mxc, mxn])
mx = torchhd.hash_table(keys, mx_values)

# combine all the associated information
mx_us = torchhd.bind(torchhd.inverse(us), mx)

# query for the dollar of mexico
usd_of_mex = torchhd.bind(mx_us, usd)

memory = torch.cat([keys, us_values, mx_values], dim=0)
print(torchhd.cosine_similarity(usd_of_mex, memory))

