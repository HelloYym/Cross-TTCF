



from bidict import bidict


a = bidict({'1':'5', '2':'6', '3':7})

print(a['1'])
print(a.inv['6'])
print(a.inv[7])